from typing import Any, Dict
import numpy as np

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
        **kwargs,
    ) -> None:
        super().__init__()
        ## falgs for MVP and maksed CNN featuers
        self.in_mask = False
        self.virtual = False
        if 'in_mask' in kwargs.keys():
            self.in_mask = kwargs['in_mask']
        if 'virtual' in kwargs.keys():
            self.virtual = kwargs['virtual']

        self.encoders = nn.ModuleDict()
        if self.in_mask:
            assert encoders.get("camera") is not None
            self.encoders["camera"] = nn.ModuleDict(
                {   
                    # "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        elif encoders.get("camera") is not None:
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
                    # "voxelize": Voxelization(**encoders["lidar"]["voxelize"]),
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

        self.init_weights()

    def init_weights(self) -> None:
        if self.in_mask:
            self.predictor = self.init_centernet()
        elif "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    ## init centernet predictor
    def init_centernet(self):
        from CenterNet2.train_net import setup
        from detectron2.engine import DefaultPredictor
        import argparse

        parser = argparse.ArgumentParser(description="CenterPoint")
        parser.add_argument('--config-file', type=str, default='configs/centernet2/nuImages_CenterNet2_DLA_640_8x.yaml')
        parser.add_argument(
            "opts",
            default=['MODEL.WEIGHTS', 'centernet2_checkpoint.pth'],
            nargs=argparse.REMAINDER,
        )
        args = parser.parse_args()
        args.opts = ['MODEL.WEIGHTS', 'centernet2_checkpoint.pth']
        cfg = setup(args)
        predictor = DefaultPredictor(cfg)
        return predictor
    
    ## post-process to remove empty mask
    def postprocess(self, res):
        result = res['instances']
        labels = result.pred_classes
        scores = result.scores 
        masks = result.pred_masks.reshape(scores.shape[0], 1600*900) 
        boxes = result.pred_boxes.tensor

        # remove empty mask and their scores / labels 
        empty_mask = (masks.sum(dim=1) == 0) & (scores > -1)

        labels = labels[~empty_mask]
        scores = scores[~empty_mask]
        masks = masks[~empty_mask]
        boxes = boxes[~empty_mask]
        masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600*900)
        # masks = masks.cpu().numpy()
        if len(masks) == 0:
            mask_feature = torch.ones([1600, 900], dtype=bool, device=masks.device)
        else:
            mask_feature = masks[0]
            for mask in masks:
                mask_feature |= mask
            mask_feature = mask_feature.reshape(1600, 900).type(torch.bool)
        return mask_feature
    
    def extract_camera_features(
        self,
        x,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        return x
    
    @torch.no_grad()
    @force_fp32()
    def extract_camera_features_in_mask(
        self,
        x,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        all_data = []
        for i in range(x.shape[0]):
            # 3 * 506 * 900
            image = x[i, :, :, :]              
            inputs = {"image": image, "height": 900, "width": 1600}
            all_data.append(inputs)
        del x
        all_data = np.stack(all_data, axis=-1)
        results, x = self.predictor.model(all_data, with_feats=True)
        for feats in x:
            feats = feats.detach()
        
        '''
        import cv2
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        for camera_id in range(6):
            im = all_data[camera_id]['image'].cpu().numpy()
            im = im.transpose([1,2,0])
            im = cv2.resize(im, (1600, 900))
            out_path = 'data/nuscenes/pics/%d' % camera_id + '.png'

            v = Visualizer(im[:,:,::-1], MetadataCatalog.get('nuimages_train'))
            out = v.draw_instance_predictions(results[camera_id]['instances'].to('cpu'))# , score_threshold=0.4)

            mask_img = out.get_image()[:,:,::-1]
            cv2.imwrite(out_path, mask_img)
            
        assert False
        '''
        # x = self.encoders["camera"]["neck"](x)
        
        if not isinstance(x, torch.Tensor):
            x = x[2]
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        
        mask_features = [self.postprocess(result) for result in results]
        masks = torch.stack(mask_features, axis=-1)
        ## BN * IW(900) * IH(1600)
        masks = masks.transpose(2, 0)

        assert masks.shape[1] == 900
        assert masks.shape[2] == 1600

        return x, masks
    
    def vtransformer(
        self,
        img_feats,
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
        masks=None,
    ) -> torch.Tensor:

        x = self.encoders["camera"]["vtransform"](
            img_feats,
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
            masks=masks,
        )
        return x
    
    # @torch.no_grad()
    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            # f, c, n = self.encoders["lidar"]["voxelize"](res)
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
            # sizes.append(n)
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        # sizes = torch.cat(sizes, dim=0)

        # if self.voxelize_reduce:
        #     feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )

        if self.virtual:
            lidar_ratio = feats[:, -1]
            mix_mask = (lidar_ratio > 0) * (lidar_ratio < 1)
            feats[mix_mask, 3] /= torch.sum(feats[mix_mask, -2:], dim=1)
            feats[mix_mask, 5:-2] /= (1 - lidar_ratio[mix_mask].unsqueeze(-1))
        
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
                metas=metas,
                img=img,
                camera2ego=camera2ego,
                lidar2camera=lidar2camera,
                lidar2image=lidar2image,
                camera_intrinsics=camera_intrinsics,
                img_aug_matrix=img_aug_matrix,
                gt_masks_bev=gt_masks_bev,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
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
        camera2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
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
                if not self.in_mask:
                    feature = self.extract_camera_features(img)
                    masks = None
                else:
                    feature, masks = self.extract_camera_features_in_mask(img)
                ## Spatial transfomration of psuedo points
                feature = self.vtransformer(
                    feature,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    masks=masks,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
            
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
