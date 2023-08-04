from typing import Any, Dict

import torch
from torch import nn
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.utils import fix_bn
from mmdet3d.models.fusers.sparse_pool import SparsePool

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)

from mmdet3d.models import FUSIONMODELS
from .base import Base3DFusionModel

__all__ = ["BEVAttn"]

@FUSIONMODELS.register_module()
class BEVAttn(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        decoder: Dict[str, Any],
        dense_head: Dict[str, Any],
        fuser: Dict[str, Any]=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mult_feat = kwargs.get('mult_feat', False)
        self.return_cl_loss = kwargs.get('return_cl_loss', False)

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": build_backbone(encoders["lidar"]["voxelize"]),
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                    "map2bev": build_backbone(encoders["lidar"]["map_to_bev_module"])
                }
            )

        self.fuser = None if fuser is None else build_fuser(fuser)
        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
            }
        )

        self.dense_head = build_head(dense_head)
        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()
                
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
        
        
        if not isinstance(x, torch.Tensor) and not self.mult_feat:
            feat_imgs = [x[0]]
        else:
            feat_imgs = x
            
        feat_bev, weights = self.encoders["camera"]["vtransform"](
            feat_imgs=feat_imgs,
            feat_pts=feat_pts,
            cam_intrinsic=camera_intrinsics,
            camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
            lidar2image=lidar2image,
            pts_ind=pts_ind,
            **kwargs,
        )
        
        return feat_bev
    
    @force_fp32()
    def voxelize(self, points):
        batch_dict = {}
        batch_dict.update(self.encoders["lidar"]["voxelize"](points))
        # self.encoders["lidar"]["backbone"](batch_dict)
        # self.encoders["lidar"]["map2bev"](batch_dict)
        return batch_dict

    @auto_fp16(apply_to=("img"))
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
            
        gt_boxes = []
        batch_size = len(gt_bboxes_3d)
        max_objs = 0
        for labels, boxes in zip(gt_labels_3d, gt_bboxes_3d):
            # boxes_ = torch.cat([boxes.tensor.to(labels.device), labels.unsqueeze(1)+1], dim=1)
            boxes_ = torch.cat([boxes.tensor[:, [0, 1, 2, 4, 3, 5, 6, 7, 8]].to(labels.device), labels.unsqueeze(1)+1], dim=1)
            boxes_[:, 6] = -boxes_[:, 6] - torch.pi/2
            boxes_[:, 6] = (boxes_[:, 6] + torch.pi) % (2 * torch.pi) - torch.pi
            gt_boxes.append(boxes_)
            max_objs = max(max_objs, boxes_.shape[0])
        
        gt_boxes_ = torch.zeros((batch_size, max_objs, gt_boxes[0].shape[-1]), dtype=gt_boxes[0].dtype, device=gt_boxes[0].device)
        for idx in range(batch_size):
            gt_boxes_[idx, 0:gt_boxes[idx].shape[0], :] = gt_boxes[idx]

        batch_dict = dict(
            batch_size=batch_size,
            gt_boxes=gt_boxes_,
            points=torch.cat([torch.cat([torch.ones(point.shape[0], 1).to(point.device) * idx, point], dim=-1) \
                for idx, point in enumerate(points)], dim=0)
        )
        
        
        batch_dict.update(self.voxelize(batch_dict['points']))
        self.encoders["lidar"]["backbone"](batch_dict)
        self.encoders["lidar"]["map2bev"](batch_dict)

        
        if "camera" in self.encoders.keys():
            pts_feat = batch_dict['spatial_features']
            pts_ind = batch_dict['spatial_masks']

            img_feat_bev = self.extract_camera_features(
                img,
                pts_feat.permute(0,1,3,2).contiguous(),
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
                # pts_ind=None,
                pts_ind=pts_ind.view(batch_size, pts_feat.shape[-2], pts_feat.shape[-1]).permute(0,2,1),
            )
            img_feat_bev = img_feat_bev.permute(0,1,3,2).contiguous()
            batch_dict.update(spatial_features_img=img_feat_bev)

            
        # import numpy as np
        # np.save('/home/kiki/jq/lss/bevfusion/visual/pts_feat', \
        #     batch_dict['spatial_features'].detach().cpu().numpy())
            
        if self.fuser is not None:
            if isinstance(self.fuser, SparsePool):
                batch_dict = self.fuser(batch_dict)
            else:
                feat_fuse = self.fuser([img_feat_bev, batch_dict['spatial_features']])
                batch_dict['spatial_features'] = feat_fuse
        
            
            # np.save('/home/kiki/jq/lss/bevfusion/visual/img_feat', \
            #     img_feat_bev.detach().cpu().numpy())
            
            # np.save('/home/kiki/jq/lss/bevfusion/visual/fuse_feat', \
            #     batch_dict['spatial_features'].detach().cpu().numpy())
        
            
            
        # feat_fuse = torch.cat([batch_dict['spatial_features'], img_feat_bev], dim=1)
        # batch_dict['spatial_features'] = feat_fuse
            
            
        self.decoder["backbone"](batch_dict)
        self.dense_head(batch_dict)
    
        if self.training:
            losses, tb_dict = self.get_training_loss(batch_dict)
            for key in tb_dict.keys():
                if not isinstance(tb_dict[key], torch.Tensor):
                    tb_dict[key] = torch.tensor(tb_dict[key], device=losses.device, requires_grad=False)
            return tb_dict
        else:
            outputs = [{} for _ in range(batch_size)]
            final_boxes = batch_dict['final_box_dicts']
            for k, final_box in enumerate(final_boxes):
                boxes = final_box['pred_boxes']
                boxes = boxes[:, [0, 1, 2, 4, 3, 5, 6, 7, 8]]
                boxes[:, 6] = -boxes[:, 6] - torch.pi/2
                boxes = LiDARInstance3DBoxes(boxes, box_dim=boxes.shape[-1])
                scores = final_box['pred_scores']
                labels = final_box['pred_labels']-1
                outputs[k].update(
                    {
                        "boxes_3d": boxes.to("cpu"),
                        "scores_3d": scores.cpu(),
                        "labels_3d": labels.cpu(),
                    }
                )

            return outputs
        
    def get_training_loss(self,batch_dict):
        
        loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict
    
    @auto_fp16(apply_to=("img"))
    def forward(
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
        **kwargs,
    ):
        
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                points,
                lidar2ego,
                lidar_aug_matrix,
                metas,
                img,
                mask,
                camera2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                metas_tta=metas_tta,
                **kwargs,
            )
            return outputs
        
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(BEVAttn, self).train(mode)
        self.apply(fix_bn)