from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
# for TTA
from mmdet3d.core.bbox import box_np_ops as box_np_ops


from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel
from .bevfusion import BEVFusion

__all__ = ["TransMVP"]


@FUSIONMODELS.register_module()
class TransMVP(BEVFusion):
    def __init__(
        self,
        encoders: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(encoders, decoder, heads)
        
        ## falgs for MVP and maksed CNN featuers
        self.virtual = False
        if 'virtual' in kwargs.keys():
            self.virtual = kwargs['virtual']

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    # "neck": build_neck(encoders["camera"]["neck"])
                }
            )
        
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)


        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    @force_fp32()
    def extract_camera_features(self, x) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        # x = self.encoders["camera"]["neck"](x)
        
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        return x

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
        **kwargs,
    ):
        features = []
        if "lidar" in self.encoders:
            features.append(self.extract_lidar_features(points))
        else:
            features.append(None)
        if "camera" in self.encoders:
            features.append(self.extract_camera_features(img))
        else:
            features.append(None)
            
        metas_align = {"lidar_aug_matrix": lidar_aug_matrix,
                       "lidar2image": lidar2image,
                       "img_aug_matrix": img_aug_matrix}
        x = features[0]
        batch_size = x.shape[0]
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, 
                                     metas,
                                     features[1], 
                                     metas_align,
                                     )
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                else:
                    raise ValueError(f"unsupported head: {type}")
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
                    pred_dict = head(x, 
                                    metas,
                                    features[1], 
                                    metas_align,
                                    )
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
            return outputs

