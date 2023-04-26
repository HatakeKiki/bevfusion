import copy
from .transfusion import TransFusionHead

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, Linear
from mmcv.runner import force_fp32
from torch import nn
from mmdet.core import bbox2roi
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor



from mmdet3d.core import (
    PseudoSampler,
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
    xywhr2xyxyr,
)
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)

__all__ = ["TwoStageTransFusionHead"]


@HEADS.register_module()
class TwoStageTransFusionHead(TransFusionHead):
    
    def _init_roi_extractor(self):
        self.fc_roi = Linear(256*7*7, 128)
        # self.fc_roi_1 = Linear(256, 128)
                
        roi_layer=dict(type='RoIAlign',
        output_size=7,
        sampling_ratio=2,
        aligned=True)
        self.bbox_roi_extractor = SingleRoIExtractor(roi_layer, 
                                                    out_channels=64, 
                                                    featmap_strides=[4])
        # self.point_linear = Linear(128, 128)
        # self.img_linear = Linear(128, 128)
        # self.output_linear = Linear(128, 128)
        # self.dropout = nn.Dropout(0.1)
        # self.norm = nn.LayerNorm(128)
        
        
        

    def forward_single(self, inputs, img_inputs, metas=None, metas_align=None):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        reference_range = inputs.shape[-2:]
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[
                :,
                8,
            ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                9,
            ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            local_max[
                :,
                1,
            ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                2,
            ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
            ..., : self.num_proposals
        ]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
            0, 2, 1
        )
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        # Transformer Decoder Layer
        # :param query: B C Pq    :param query_pos: B Pq 3/6
        query_feat = self.decoder[-1](
            query_feat, lidar_feat_flatten, query_pos, bev_pos, reference_range=reference_range,
        )

        ret_dicts = []
        # Prediction
        res_layer = self.prediction_heads[0](query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)

        
        if self.roi_extract:
            # get pred boxes, carefully ! donot change the network outputs
            score = copy.deepcopy(res_layer["heatmap"].detach())
            center = copy.deepcopy(res_layer["center"].detach())
            height = copy.deepcopy(res_layer["height"].detach())
            dim = copy.deepcopy(res_layer["dim"].detach())
            rot = copy.deepcopy(res_layer["rot"].detach())
            vel = None

            boxes_dict = self.bbox_coder.decode(
                score, rot, dim, center, height, vel
            )  # decode the prediction to real world metric bbox
            
            list_of_img_input = []
            list_of_metas_align = []
            for batch_idx in range(batch_size):
                align = {}
                for key in metas_align.keys():
                    align[key] = metas_align[key][batch_idx, :]
                list_of_img_input.append(img_inputs[batch_idx, :])
                list_of_metas_align.append(align)
            
            # list-len: batch_size
            # [1200, 256, 7, 7]
            # [200, 12544]
            # [4, 200, 128]
            list_bboxes_feats_img = multi_apply(self.fuse_img, boxes_dict, list_of_img_input, list_of_metas_align)
            bboxes_feats_img = torch.stack(list_bboxes_feats_img[0], dim=0).permute(0, 2, 1)
            
            
            # TODO: concatenate
            
            query_feat = torch.cat([query_feat, bboxes_feats_img], dim=1)
            # Prediction
            res_layer = self.prediction_heads[-1](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            
        
        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        res_layer["dense_heatmap"] = dense_heatmap
        ret_dicts.append(res_layer)
        
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                new_res[key] = ret_dicts[0][key]
                
        
        return [new_res]
    
    def fuse_img(self, boxes_dict, img_input, metas_align):

        boxes_dict["bboxes"][..., 2] -= boxes_dict["bboxes"][..., 5] / 2
        boxes = LiDARInstance3DBoxes(boxes_dict["bboxes"], box_dim=7)

        corners = boxes.corners
        num_bboxes = corners.shape[0]

        
        coords = torch.cat(
            [corners.reshape(-1, 3), corners.new_ones(size=(num_bboxes*8, 1))], dim=-1
        )
        coords = coords @ torch.inverse(metas_align['lidar_aug_matrix'].T)
        
        
        num_view = metas_align['lidar2image'].shape[0]
        list_lidar2image = []
        list_img_aug_matrix = []
        list_coords = []
        list_img_input = []
        
        for i in range(num_view):
            list_lidar2image.append(metas_align['lidar2image'][i, :])
            list_img_aug_matrix.append(metas_align['img_aug_matrix'][i, :])
            list_coords.append(coords)
            list_img_input.append(img_input[i, :])
        
        ret = multi_apply(self.fuse_img_single, list_coords, list_lidar2image, list_img_aug_matrix)
        list_bboxes_2d, list_on_img = ret
        on_img_count = torch.stack(list_on_img, dim=1)
        on_img_count = on_img_count.sum(dim=1).reshape(-1, 1)

        on_img_indices = torch.cat(list_on_img, dim=0)
        
        rois = bbox2roi(list_bboxes_2d)
        
        # [num_proposal*num_view, 256, 7, 7]
        bbox_feats = self.bbox_roi_extractor(img_input.unsqueeze(0), rois)
        
        num_bbox = bbox_feats.shape[0]
        bbox_feats = bbox_feats.reshape(num_bbox, -1)
        # bbox_feats[~on_img_indices] = 0
        
        
        
        bbox_feats = bbox_feats.reshape(self.num_proposals, num_view, -1)
        on_img_indices = on_img_indices.reshape(self.num_proposals, num_view, -1)
        bbox_feats_sum = torch.zeros_like(bbox_feats[:self.num_proposals, 0, :])
        for i in range(self.num_proposals):
            for j in range(num_view):
                if on_img_indices[i, j, 0]:
                    bbox_feats_sum[i, :] = bbox_feats_sum[i, :] + bbox_feats[i, j, :]
        # / on_img_count
        assert bbox_feats_sum.isnan().sum() == 0
        on_img_count = torch.clamp(on_img_count, min=1.0)
        
        # bbox_feats_sum = self.fc_roi_1(self.fc_roi_0(bbox_feats_sum / on_img_count))
        bbox_feats_sum = self.fc_roi(bbox_feats_sum / on_img_count)
        
        assert bbox_feats_sum.isnan().sum() == 0

        return [bbox_feats_sum]

    def fuse_img_single(self, coords, lidar2image, img_aug_matrix):
        # lidar2image
        coords = coords @ lidar2image.T
        coords = coords.reshape(-1, 8, 4)
        
        indices = torch.all(coords[..., 2] > 0, dim=1)
        # coords = coords[indices]
        
        coords = coords.reshape(-1, 4)
        coords[:, 2] = torch.clamp(coords[:, 2], min=1e-5, max=1e4)
        
        # get 2d coords
        # dist = coords[:, 2]
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        
        # imgaug
        coords = coords @ img_aug_matrix.T
        coords = coords[..., :2].reshape(-1, 8, 2)
        
        
        coords_2d = torch.stack([coords[:, :, 0].min(dim=1)[0].clamp(0),
                                 coords[:, :, 1].min(dim=1)[0].clamp(0),
                                 coords[:, :, 0].max(dim=1)[0].clamp(0, 704),
                                 coords[:, :, 1].max(dim=1)[0].clamp(0, 256)], dim=1)
        
        on_img = (coords_2d[:, 2] > coords_2d[:, 0]) & (coords_2d[:, 3] > coords_2d[:, 1]) & indices
        coords_2d[:, 2] = torch.max(coords_2d[:, 2], coords_2d[:, 0] + 0.01)
        coords_2d[:, 3] = torch.max(coords_2d[:, 3], coords_2d[:, 1] + 0.01)
        
        # import numpy as np
        # np.save('visual/coords', coords.detach().cpu().numpy())
        
        return coords_2d, on_img
    
    def forward(self, lidar_feats, metas, img_feats=None, metas_align=None):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if isinstance(lidar_feats, torch.Tensor):
            lidar_feats = [lidar_feats]
        if isinstance(img_feats, torch.Tensor):
            img_feats = [img_feats]
        res = multi_apply(self.forward_single, lidar_feats, img_feats, [metas], [metas_align])
        assert len(res) == 1, "only support one level features."
        return res