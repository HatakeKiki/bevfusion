from typing import Any, Dict

import torch
from torch import nn
from mmcv.runner import auto_fp16, force_fp32
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from mmcv.cnn.bricks.activation import build_activation_layer
from mmdet.core import multi_apply
import numpy as np


from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet.models.builder import build_head as build_head_2d

from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from mmdet3d.models.fusion_models.bevfusion import BEVFusion

__all__ = ["BEVAttnV2"]

@FUSIONMODELS.register_module()
class BEVAttnV2(BEVFusion):
    def __init__(
        self,
        encoders: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        fuser: Dict[str, Any]=None,
        return_cl_loss: bool=False,
        mult_feat: bool=False,
        heads_extra: bool=False,
        perspective_head: Dict[str, Any]=None,
        **kwargs,
    ) -> None:
        super().__init__(encoders, decoder, heads, fuser, **kwargs)
        
        self.with_mask = kwargs.get('with_mask', False)
            
        
        self.mult_feat = mult_feat
        self.return_cl_loss = return_cl_loss
        # self.activate = nn.ReLU(inplace=True)
        # self.fuser = None
        
        if perspective_head is not None:
            self.perspective_head = build_head_2d(perspective_head)
        else:
            self.perspective_head = None
        # if heads_extra:
        #     self.heads_extra = nn.ModuleDict()
        #     for name in heads:
        #         if name == 'object':
        #             heads[name]['in_channels'] = int(heads[name]['in_channels']/2)
        #         if heads[name] is not None:
        #             self.heads_extra[name] = build_head(heads[name])
        # else:
        #     self.heads_extra = None
                
    
    def extract_camera_features(
        self,
        img,
        feat_pts,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        points_single=None,
        mask=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        pts_ind=None,
        **kwargs,
    ) -> torch.Tensor:
        
        
        x = img
        B, N, C, img_h, img_w = x.size()
        x = x.view(B * N, C, img_h, img_w)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        
        if self.perspective_head is not None:
            persp_pred = self.perspective_head(x)
            _, _, centerness = persp_pred
        else:
            centerness = None
            persp_pred = None
        
        if not isinstance(x, torch.Tensor) and not self.mult_feat:
            feat_imgs = [x[0]]
        else:
            feat_imgs = x
        
        
        if self.return_cl_loss and self.training:
            loss_cl = dict()
            depth_gt = self.get_depth_gt(points_single, img, 
                                        img_aug_matrix, 
                                        lidar_aug_matrix, 
                                        lidar2image)
            
        feat_bev, weights = self.encoders["camera"]["vtransform"](
            feat_imgs=feat_imgs,
            feat_pts=feat_pts,
            cam_intrinsic=camera_intrinsics,
            camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
            lidar2image=lidar2image,
            scores=centerness,
            pts_ind=pts_ind,
            **kwargs,
        )
        
        # if self.return_cl_loss and self.training:
        #     loss = self.encoders["camera"]["vtransform"].get_cl_loss(depth_gt, weights, int(img_h / H))
        #     loss_cl.update({'loss_cl_%d' % img_downsample_factor:loss})
        
        if self.return_cl_loss and self.training:
            return feat_bev, persp_pred, loss_cl
        else:
            return feat_bev, persp_pred
    
    @force_fp32()
    def get_depth_gt(self, points, img, img_aug_matrix, lidar_aug_matrix, lidar2image):
        batch_size, N, C, img_h, img_w = img.size()
            
        # [bs, N, img_h, img_w]
        depth_gt = torch.zeros(batch_size, N, img_h, img_w).to(
            points[0].device
        )
        
        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e4)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < img_h)
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < img_w)
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth_gt[b, c, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
        
        return depth_gt
    
    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        points,
        lidar2ego,
        lidar_aug_matrix,
        metas,
        img=None,
        mask=None,
        camera2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        metas_tta=None,
        points_single=None,
        **kwargs,
    ):
        features = []
        sensors = self.encoders if self.training else list(self.encoders.keys())[::-1]
        if "lidar" in sensors:
            pts_feat, pts_ind = self.extract_lidar_features(points)
            features.append(pts_feat)
            
        else:
            pts_feat = None

        
        if "camera" in sensors:
            tmp = self.extract_camera_features(
                img,
                pts_feat,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                points_single=points_single,
                mask=mask,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                pts_ind=pts_ind,
            )

            img_feat_bev, persp_pred = tmp

            features.append(img_feat_bev)
        
        
        features = features[::-1]
            
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
            
        # import numpy as np
        # np.save('/home/kiki/jq/lss/bevfusion/visual/pts_feat_2', \
        #     pts_feat.detach().cpu().numpy())
        
        # np.save('/home/kiki/jq/lss/bevfusion/visual/img_feat_2', \
        #     img_feat_bev.detach().cpu().numpy())
        
        # np.save('/home/kiki/jq/lss/bevfusion/visual/fuse_feat_2', \
        #     x.detach().cpu().numpy())
            
        batch_size = x.shape[0]
        
        # x = pts_feat
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        
        
        
        # if self.fuser is not None:
        #     x = [self.fuser([x[0], img_feat_bev])]
            
        # x = [img_feat_bev]
            
        # x = torch.cat([x[0], img_feat_bev], dim=1)
        # else:
        #     assert len(features) == 1, features
        #     x = features[0]
        
        if self.training:
            outputs = {}
            ## detection head
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                else:
                    raise ValueError(f"unsupported head: {type}")
            ## depth supervision
            # if self.return_cl_loss:
            #     losses.update(loss_cl)
                
            ## perspetive head     
            if self.perspective_head is not None:
                list_bboxes_2d = []
                list_labels_2d = []
                bboxes_2d, labels_2d = multi_apply(self.lidar2img, 
                                                    gt_bboxes_3d, 
                                                    gt_labels_3d, 
                                                    lidar_aug_matrix,
                                                    lidar2image,
                                                    img_aug_matrix)
                
                for i in range(batch_size):
                    list_bboxes_2d += bboxes_2d[i]
                    list_labels_2d += labels_2d[i]
                    
                if self.with_mask:
                    # B, N, C, H, W =  img.shape
                    # mask = mask.view(B * N, C, H, W)
                    # mask =[self.mask_downsample(mask[:, 0, ...], down_factor=i).reshape(-1) for i in [8, 16]]
                    mask = None
                else:
                    mask = None
                    
                    
                loss_perspective = self.perspective_head.loss(persp_pred[0], 
                                                              persp_pred[1],
                                                              persp_pred[2],
                                                              list_bboxes_2d, 
                                                              list_labels_2d,
                                                              mask=mask,
                                                              img_metas=None)
                losses.update(loss_perspective)
            
            # metas_align = {"lidar_aug_matrix": lidar_aug_matrix,
            #                 "lidar2image": lidar2image,
            #                 "img_aug_matrix": img_aug_matrix,
            #                 "bboxes_2d": list_bboxes_2d,
            #                 "labels_2d": list_labels_2d,
            #                 "bboxes_3d": gt_bboxes_3d,
            #                 "labels_3d": gt_labels_3d}
            
            # import pickle
            # np.save('/home/kiki/jq/lss/bevfusion/visual/fcos_img', img.detach().cpu().numpy())
            # np.save('/home/kiki/jq/lss/bevfusion/visual/fcos_mask', mask.detach().cpu().numpy())
            # np.save('/home/kiki/jq/lss/bevfusion/visual/fcos_pts', points[0].detach().cpu().numpy())
            # with open("/home/kiki/jq/lss/bevfusion/visual/fcos_metas", 'wb') as f:
            #     pickle.dump(metas_align, f)

            ## loss print in log
            for name, val in losses.items():
                if val.requires_grad:
                    outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                else:
                    outputs[f"stats/{type}/{name}"] = val
                        
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
                
            
            return outputs
        
    def lidar2img(self, 
                  gt_bboxes_3d, 
                  gt_labels_3d, 
                    lidar_aug_matrix,
                    lidar2image,
                    img_aug_matrix):
        
        # Reverse LiDAR augmentation
        corners = gt_bboxes_3d.corners.to(lidar_aug_matrix.device)
        height = gt_bboxes_3d.height[:, None]
        corners[..., -1] -= height.to(lidar_aug_matrix.device) / 2
        num_bboxes = corners.shape[0]
        
        coords = torch.cat(
            [corners.reshape(-1, 3), corners.new_ones(size=(num_bboxes*8, 1))], dim=-1
        )
        coords = coords @ torch.inverse(lidar_aug_matrix.T)
        
        # projection to image plane, then replay image augmentation
        num_view = lidar2image.shape[0]
        list_lidar2image = []
        list_img_aug_matrix = []
        list_coords = []
        list_labels = []

        for i in range(num_view):
            list_lidar2image.append(lidar2image[i, :])
            list_img_aug_matrix.append(img_aug_matrix[i, :])
            list_coords.append(coords)
            list_labels.append(gt_labels_3d.to(lidar_aug_matrix.device))
        
        list_bboxes_2d, list_labels_2d = multi_apply(self.lidar2img_single, list_coords, list_labels, list_lidar2image, list_img_aug_matrix)
        
        # on_img_count = torch.stack(list_on_img, dim=1)
        # on_img_count = on_img_count.sum(dim=1).reshape(-1, 1)

        # on_img_indices = torch.cat(list_on_img, dim=0)
        
        return list_bboxes_2d, list_labels_2d

    def lidar2img_single(self, coords, labels, lidar2image, img_aug_matrix):
        
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
        
        labels_2d = labels[on_img]
        coords_2d = coords_2d[on_img, :]
        coords_2d[:, 2] = torch.max(coords_2d[:, 2], coords_2d[:, 0] + 0.01)
        coords_2d[:, 3] = torch.max(coords_2d[:, 3], coords_2d[:, 1] + 0.01)
        
        
        return coords_2d, labels_2d
    
    
    def mask_downsample(self, mask, down_factor):
        # mask_new = torch.ones((mask.shape[0], int(mask.shape[1]/down_factor), int(mask.shape[2]/down_factor)), device=mask.device, requires_grad=False)
        # # mask_new = np.ones((mask.shape[0], int(mask.shape[1]/down_factor), int(mask.shape[2]/down_factor)))
        # for i in range(mask_new.shape[1]):
        #     for j in range(mask_new.shape[2]):
        #         patch = mask[:, i*down_factor:(i+1)*down_factor, j*down_factor:(j+1)*down_factor]
        #         patch = patch.reshape(mask_new.shape[0], -1)
        #         sum_val = patch.sum(dim=1)
        #         for idx in range(mask_new.shape[0]):
        #             if sum_val[idx] < 1:
        #                 mask_new[idx, i, j] = 0
        with torch.no_grad():
            bool_down_sample = nn.MaxPool2d(kernel_size=(down_factor, down_factor), stride=(down_factor, down_factor))
            mask_new = bool_down_sample(mask)
                    
        return mask_new