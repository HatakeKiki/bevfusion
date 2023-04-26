from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import int32, nn
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        bevpool_v2: bool = False,
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            bevpool_v2=bevpool_v2,
        )
        # TODO: Lidar points for depth
        # self.dtransform = nn.Sequential(
        #     nn.Conv2d(1, 8, 1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 32, 5, stride=4, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        # )
        # self.depthnet = nn.Sequential(
        #     nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(in_channels, in_channels, 3, padding=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(in_channels, self.D + self.C, 1),
        # )
        
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.C, 1),
        )
                
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        # TODO: Lidar points for depth
        # d = self.dtransform(d)
        # x = torch.cat([d, x], dim=1)
        # x = self.depthnet(x)
        # depth = x[:, : self.D].softmax(dim=1)
        # x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        

        # Uniform depth distribution
        x = self.depthnet(x)
        depth = torch.ones((x.shape[0], self.D, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device)
        x = depth.unsqueeze(1) * x.unsqueeze(2)
        del depth
        depth = None
        
        
        
        # save_d = depth.cpu().numpy()
        # import numpy as np
        # np.save('/home/kiki/jq/MVP/depth_bevc.npy', save_d)
        

        # Predicted ray distribution
        # depth = x[:, :1].softmax(dim=1)
        # depth_list = []
        # for i in range(118):
        #     depth_list.append(depth)
        # depth = torch.stack(depth_list, dim=2)
        # x = depth * x[:, 1 : (1 + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth
    
    @force_fp32()
    def get_cam_feats_depth(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        # TODO: Lidar points for depth
        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        feats = x[:, self.D : (self.D + self.C)]

        depth = depth.view(B, N, self.D, fH, fW)
        feats = feats.view(B, N, self.C, fH, fW)
        return feats, depth
    
    def voxel_pooling_prepare_v2(self, coor, out_bool=None):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range( # repeative ranks of image features (B*N*H*W)
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        
        # convert coordinate into the voxel space
        coor = ((coor - (self.bx - self.dx / 2.0)) / self.dx).long()
        coor = coor.long().view(num_points, 3)
        
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.nx[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.nx[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.nx[2])
               
        # filter out points that are outside mask
        if out_bool is not None:
            # assert False
            mask_bool = []
            for i in range(D):
                mask_bool.append(out_bool)
            mask_bool = torch.stack(mask_bool, axis=1)
            mask_bool = mask_bool.view(num_points).bool()
            if (mask_bool).sum() == 0:
                mask_bool[0] = True
            kept = kept & mask_bool
               
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        
        ranks_bev = (
            coor[:, 3] * (self.nx[2] * self.nx[1] * self.nx[0])
            + coor[:, 2] * (self.nx[1] * self.nx[0])
            + coor[:, 1] * self.nx[0]
            + coor[:, 0]
        )
        
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()
        
    def voxel_pooling_v2(self, coor, depth, feat, benchmark_vt=False, out_bool=None):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor, out_bool=out_bool)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                    'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.nx[2]),
                int(self.nx[0]),
                int(self.nx[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.nx[2]),
                            int(self.nx[1]), int(self.nx[0]),
                            feat.shape[-1])  # (B, Z, Y, X, C)
        if benchmark_vt:
            return depth, feat, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts, interval_lengths
        import numpy as np
        np.save('/home/kiki/jq/lss/bevfusion/depth_attn/ranks_feat', \
            ranks_feat.detach().cpu().numpy())
        np.save('/home/kiki/jq/lss/bevfusion/depth_attn/ranks_bev', \
            ranks_bev.detach().cpu().numpy())
        np.save('/home/kiki/jq/lss/bevfusion/depth_attn/interval_starts', \
            interval_starts.detach().cpu().numpy())
        np.save('/home/kiki/jq/lss/bevfusion/depth_attn/interval_lengths', \
            interval_lengths.detach().cpu().numpy())
        
        a = interval_lengths.max()
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                                bev_feat_shape, interval_starts,
                                interval_lengths)
        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        if 'benchmark_vt' in kwargs.keys() and kwargs['benchmark_vt'] == True:
            return x
        x = self.downsample(x)
        return x