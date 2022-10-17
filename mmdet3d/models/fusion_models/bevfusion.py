from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

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

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        semantic: Dict[str, Any]=None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        ## falgs for MVP and maksed CNN featuers
        self.with_mask = False
        self.virtual = False
        self.middle_fuse = False
        if 'with_mask' in kwargs.keys():
            self.with_mask = kwargs['with_mask']
        if 'virtual' in kwargs.keys():
            self.virtual = kwargs['virtual']

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
        # '''
        if semantic is not None:
            if semantic["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**semantic["voxelize"])
            else:
                voxelize_module = DynamicScatter(**semantic["voxelize"])
            self.semantic = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(semantic["backbone"]),
                }
            )
            self.middle_fuse = True
        # '''
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

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
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
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        

        if not isinstance(x, torch.Tensor):
            x = x[0]
            
        if self.with_mask:
            mask = mask.view(B * N, C, H, W)
            out_sem = self.semantic_downsample(mask, with_bool=False)
            x = torch.cat([x, out_sem.detach()], dim=1)
            del out_sem

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def semantic_downsample(self, x, with_bool=False, down_factor=8, class_num=10):
        sem_bool = x[:, 2, :, :]
        sem_score = x[:, 0, :, :]
        sem_label = x[:, 1, :, :]
        
        sem_down_sample = nn.AvgPool2d(kernel_size=(down_factor, down_factor), stride=(down_factor, down_factor))

        cats = torch.tensor([i+1 for i in range(class_num)])/255

        BN, W, H = sem_label.shape
        out_sem_mask = torch.zeros([BN*W*H, class_num], dtype=torch.float32, device=x.device)

        sem_label_ = sem_label.reshape(-1)
        sem_score_ = sem_score.reshape(-1)
        for cat in cats:
            index = (sem_label_ == cat)
            out_sem_mask[index, int(cat*255) - 1] = sem_score_[index]
        out_sem_mask = out_sem_mask.reshape(BN, W, H, class_num).permute(0, 3, 1, 2)
        out_sem_mask = sem_down_sample(out_sem_mask)
        
        if with_bool:
            bool_down_sample = nn.MaxPool2d(kernel_size=(down_factor, down_factor), stride=(down_factor, down_factor))
            out_bool = bool_down_sample(sem_bool)
            return out_bool, out_sem_mask
        else:
            return out_sem_mask

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    def extract_semantic_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.semantic_voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.semantic["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
            # feats = feats.contiguous()
            
        if self.virtual:
            lidar_ratio = feats[:, -1]
            mix_mask = (lidar_ratio > 0) * (lidar_ratio < 1)
            feats[mix_mask, 3] /= torch.sum(feats[mix_mask, -2:], dim=1)
            feats[mix_mask, 5:-2] /= (1 - lidar_ratio[mix_mask].unsqueeze(-1))
        
        feats = feats.contiguous()
        
        return feats, coords, sizes
    @torch.no_grad()
    @force_fp32()
    def semantic_voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.semantic["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
            feats = feats.contiguous()
        
        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
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
                **kwargs,
            )
            return outputs

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
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    mask
                )
            elif sensor == "lidar":
                if self.middle_fuse:
                    lidar_points = []
                    virtual_points = []
                    for bs in range(len(points)):
                        lidar_mask = (points[bs][:, -1] == 1)
                        lidar_points.append(points[bs][lidar_mask, 0:5])
                        virtual_points.append(points[bs][~lidar_mask, 0:17])
                    semantic_feature = self.extract_semantic_features(virtual_points)
                else:
                    semantic_feature = None
                    lidar_points = points
                feature = self.extract_lidar_features(lidar_points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
            if (sensor == 'lidar') and (semantic_feature is not None):
                features.append(semantic_feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
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
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
