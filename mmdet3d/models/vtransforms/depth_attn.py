from mmdet3d.models.builder import VTRANSFORMS
from typing import Tuple
import torch
from torch import nn
from mmcv.runner import force_fp32
from mmdet3d.models.vtransforms.base import BaseTransform
from mmdet3d.modules.depth_attn_layer import DepthAttnLayer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.cnn.bricks.transformer import build_attention



__all__ = ["DepthAttnTransform"]

@VTRANSFORMS.register_module()
class DepthAttnTransform(BaseTransform):
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
        depth_attn_layers: int = 1,
        depth_attn_cfg: dict = None,
        positional_encoding_key=None,
        positional_encoding=None,
        downsample: int=2,
        with_bev_embedding=True,
        **kwargs,
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
        )
        
        self.with_bev_embedding = with_bev_embedding
        self.in_channels = in_channels
        
        if self.with_bev_embedding:
            self.bev_embedding = nn.Embedding(
                    self.nx[0] * self.nx[1], in_channels)
        
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(positional_encoding)
        else:
            self.positional_encoding = None
            
        if positional_encoding_key is not None:
            self.positional_encoding_key = build_positional_encoding(positional_encoding_key)
        else:
            self.positional_encoding_key = None

        pos_bev = self.create_pos_bev(self.nx[0], self.nx[1]).transpose(2, 3)
        self.pos_bev = pos_bev.contiguous().view(1, 2, -1)
        
        self.depth_attn = nn.ModuleList()
        for i in range(depth_attn_layers):
            self.depth_attn.append(
                build_attention(depth_attn_cfg)
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
    
    def create_pos_bev(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x_, batch_y_ = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        # x(down), y(right) -> x(up), y(left)
        batch_x = -(batch_x_ + 0.5 - x_size / 2) * self.xbound[-1]
        batch_y = -(batch_y_ + 0.5 - y_size / 2) * self.ybound[-1]
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        # print(batch_x.shape)
        return coord_base[None]
    
    def get_ranks(self, coor):
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
        # coor of pseudo points
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        # repeative ranks of image features (B*N*H*W)
        ranks_feat = torch.range( 
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
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_feat = coor[kept], ranks_feat[kept]
        
        # bev index
        ranks_bev = (
            coor[:, 3] * (self.nx[2] * self.nx[1] * self.nx[0])
            + coor[:, 2] * (self.nx[1] * self.nx[0])
            + coor[:, 1] * self.nx[0]
            + coor[:, 0]
        )
        
        # in feat order
        order = ranks_feat.argsort()
        ranks_bev, ranks_feat = ranks_bev[order], ranks_feat[order]

        kept = torch.ones(
            ranks_feat.shape[0], device=ranks_feat.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        
        return ranks_bev.int().contiguous(), ranks_feat.int().contiguous(), \
               interval_starts.int().contiguous(), interval_lengths.int().contiguous()
    
    
    @force_fp32()
    def forward(
        self,
        feat_img,
        feat_pts,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        return_ranks=False,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]

        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            # img aug
            post_rots,
            post_trans,
            # lidar aug
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        ranks_bev, ranks_feat, interval_starts, interval_lengths = self.get_ranks(geom)
        
        if return_ranks:
            return ranks_bev, ranks_feat, interval_starts, interval_lengths
        
        
        bs, N, C, _, _ = feat_img.shape
        output_shape = (bs, int(self.nx[0]), int(self.nx[1]), C)
        
        if self.with_bev_embedding:
            bev_queries = self.bev_embedding.weight.to(feat_img.dtype).view(self.nx[0], self.nx[1], C).permute(2, 0, 1).unsqueeze(0)
            query_pos = bev_queries.repeat(bs, 1, 1, 1).permute(0, 2, 3, 1).contiguous().view(-1, C)
        else:
            query_pos=None
        
        if self.positional_encoding is not None:
            query_depth = self.positional_encoding(self.pos_bev.to(feat_img.device))
            query_depth = query_depth.repeat(bs, 1, 1).permute(0, 2, 1).contiguous().view(-1, C)
        else:
            assert self.with_bev_embedding
            query_depth = torch.zeros_like(query_pos)
        
        if self.positional_encoding_key is not None:
            mask = torch.zeros(feat_img.shape[-2:], device=feat_img.device).unsqueeze(0)
            key_pos = self.positional_encoding_key(mask)
            key_pos = key_pos.repeat(bs*N, 1, 1, 1)
            key_pos = key_pos.permute(0, 2, 3, 1).contiguous().view(-1, C)
        else:
            key_pos = None
        # [bs, N, C, h, w] -> [bs, N, h, w, C]
        key = feat_img.permute(0, 1, 3, 4, 2).contiguous().view(-1, C)
        value = key
        
        if feat_pts is not None:
            # [bs, C, H, W] -> [bs, W, H, C]
            query_sem = feat_pts.permute(0, 3, 2, 1).contiguous().view(-1, C)
        else:
            query_sem = None
        
        
        for layer in self.depth_attn:
            output = layer(query_depth=query_depth, query_sem=query_sem, key=key, value=value, 
                           ranks_feat_f=ranks_feat, ranks_bev_f=ranks_bev, 
                           interval_starts_f=interval_starts, interval_lengths_f=interval_lengths,
                           output_shape=output_shape, query_pos=query_pos, key_pos=key_pos)
            query_depth = output
        
        # Reverse W and H
        # [bs, W, H, C] -> [bs, H, W, C]
        x = query_depth.view(output_shape)
        x = x.permute(0, 3, 2, 1)
        x = self.downsample(x)
        
        
        return x