from typing import Any, Dict

import torch
from torch import nn
from mmcv.runner import auto_fp16, force_fp32

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from mmdet3d.models.fusion_models.bevfusion import BEVFusion

__all__ = ["BEVAttn"]


@FUSIONMODELS.register_module()
class BEVAttn(BEVFusion):
    def __init__(
        self,
        encoders: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        fuser: Dict[str, Any]=None,
        **kwargs,
    ) -> None:
        super().__init__(encoders, decoder, heads)

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

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

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
                    
        self.virtual = False

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
        mask=None,
        **kwargs,
    ) -> torch.Tensor:
        x = img
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        
        if not isinstance(x, torch.Tensor):
            x = x[0]
            
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        
        x = self.encoders["camera"]["vtransform"](
            feat_img=x,
            feat_pts=feat_pts,
            cam_intrinsic=camera_intrinsics,
            camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
        )
        return x