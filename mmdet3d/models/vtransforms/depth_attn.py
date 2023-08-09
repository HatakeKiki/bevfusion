from mmdet3d.models.builder import VTRANSFORMS
from typing import Tuple
import torch
from torch import nn
from mmcv.runner import force_fp32
from mmdet3d.models.vtransforms.base import BaseTransform
from mmdet3d.modules.depth_attn_layer import DepthAttnLayer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.cnn import build_norm_layer, build_activation_layer

import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.init import normal_


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
        # positional_encoding_key=None,
        positional_encoding=None,
        linear_layer=None,
        downsample: int=1,
        with_bev_embedding=True,
        img_downsample_factor=[1],
        loss_cl: dict = dict(loss_weight=3.0,
                              T = 0.07),
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
        assert img_downsample_factor[0] == 1
        self.with_bev_embedding = with_bev_embedding
        self.mult = len(img_downsample_factor) > 1
        self.loss_cl_weight = kwargs.get('loss_weight', 1)
        
        # if (self.mult):
        #     self.level_embeds = nn.Parameter(torch.Tensor(len(img_downsample_factor), self.in_channels))
        #     normal_(self.level_embeds)

        if linear_layer is not None:
            self.linear_layer = nn.Linear(linear_layer['input_channel'], linear_layer['output_channel'])
            # self.out_linear = nn.Linear(linear_layer['output_channel'], linear_layer['input_channel'])
        else:
            self.linear_layer = None
        # self.linear_layer = None
        # self.out_proj = nn.Linear(256, 512)
        
            
        self.frustums = nn.ParameterList()
        for factor in img_downsample_factor:
            if factor == 1:
                self.frustums.append(self.frustum)
                self.frustum = None
            else:
                self.feature_size = [int(feature_size[0] / factor), int(feature_size[1] / factor)]
                self.frustums.append(self.create_frustum())
                
        self.feature_size = feature_size
        
        
        if self.with_bev_embedding:
            self.bev_embedding = nn.Embedding(self.nx[0] * self.nx[1], in_channels)

        
        if positional_encoding is not None:
            pos_bev = self.create_pos_bev(self.nx[0], self.nx[1], self.xbound[-1], self.ybound[-1]).transpose(2, 3)
            self.pos_bev = nn.Parameter(pos_bev.contiguous().view(1, 2, -1), requires_grad=False)
            self.positional_encoding = build_positional_encoding(positional_encoding)
        else:
            self.positional_encoding = None
        
        # if not self.mult:
        self.score_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 1, 1),
        )
        
        self.depth_attn = nn.ModuleList()
        for i in range(depth_attn_layers):
            self.depth_attn.append(
                build_attention(depth_attn_cfg)
            )
    
        self.loss_cl = loss_cl
        self.criterion = nn.CrossEntropyLoss()
        
        if downsample > 1:
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
            self.downsample = None
        
    def create_pos_bev(self, x_size, y_size, x_step, y_step):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x_, batch_y_ = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        # x(down), y(right) -> x(up), y(left)
        batch_x = -(batch_x_ + 0.5 - x_size / 2) * x_step
        batch_y = -(batch_y_ + 0.5 - y_size / 2) * y_step
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        # print(batch_x.shape)
        return coord_base[None]
    
    def get_ranks(self, coor, bx, dx, nx):
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
        coor = ((coor - (bx - dx / 2.0)) / dx).long()
        coor = coor.long().view(num_points, 3)
        
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < nx[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < nx[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < nx[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_feat = coor[kept], ranks_feat[kept]
        
        # bev index
        ranks_bev = (
            coor[:, 3] * (nx[2] * nx[1] * nx[0])
            + coor[:, 2] * (nx[1] * nx[0])
            + coor[:, 1] * nx[0]
            + coor[:, 0]
        )
        
        return ranks_bev.int().contiguous(), ranks_feat.int().contiguous()
    
    def get_ranks_ordered(self, ranks_bev, ranks_feat):
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
        
        # bev_idx = (~kept).to(torch.int32)
        # offset = torch.cat([torch.tensor([1]).to(interval_lengths.device), interval_lengths[:-1]], dim=0).to(torch.int32)
        # bev_idx[bev_idx == 0] = -offset+1
        # bev_idx = bev_idx.cumsum(0)
        
        return ranks_bev.int().contiguous(), ranks_feat.int().contiguous(), \
               interval_starts.int().contiguous(), interval_lengths.int().contiguous()
             
    @force_fp32()
    def forward(
        self,
        feat_imgs,
        feat_pts,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        scores=None,
        pts_ind=None,
        sparse_bev_ind=None,
        sparse_per_ind=None,
        **kwargs,
    ):
        assert len(feat_imgs) == len(self.frustums)
        
        # pseudo points from image plane to lidar coordinate
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        
        ranks_bev = []
        ranks_feat = []
        for idx, frustum in enumerate(self.frustums):
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
                frustum=frustum,
            )
            tmp = self.get_ranks(geom, self.bx, self.dx, self.nx)
            ranks_bev.append(tmp[0])
            ranks_feat.append(tmp[1])

        BN, c_img, _, _ = feat_imgs[0].shape
        
        if feat_pts is None:
            feat_pts = torch.zeros((int(BN/6), c_img, self.nx[0], self.nx[1])).to(feat_imgs[0].device)
            
        bs, c_pts, H, W = feat_pts.shape
        output_shape = (bs, W, H, c_img)
        
        
        offset=0
        for idx, feat_img in enumerate(feat_imgs[:-1]):
            BN, _, h, w = feat_img.shape
            offset += BN*h*w 
            ranks_feat[idx+1] = ranks_feat[idx+1] + offset

        ranks_bev = torch.cat(ranks_bev, dim=0)
        ranks_feat = torch.cat(ranks_feat, dim=0)
        
        # sparse ranks
        if pts_ind is not None:
            pts_ind = pts_ind.permute(0, 2, 1).contiguous()
            pts_ind = pts_ind.view(-1).to(torch.bool)
            sparse_ind = pts_ind[ranks_bev.to(torch.int64)]
            ranks_bev = ranks_bev[sparse_ind]
            ranks_feat = ranks_feat[sparse_ind]
        
        # in feat order
        ranks_bev, ranks_feat, interval_starts, interval_lengths = \
                                self.get_ranks_ordered(ranks_bev, ranks_feat)
                                        

        key = torch.cat([feat_img.permute(0, 2, 3, 1).contiguous().view(-1, c_img) for feat_img in feat_imgs], dim=0).contiguous()

        if scores is not None:
            feat_imgs = [feat_img * score.sigmoid().expand(-1, self.in_channels, -1, -1) for score, feat_img in zip(scores, feat_imgs)]
        else:
            feat_imgs = [feat_img * self.score_net(feat_img).sigmoid().expand(-1, self.in_channels, -1, -1) for feat_img in feat_imgs]
        value = torch.cat([feat_img.permute(0, 2, 3, 1).contiguous().view(-1, c_img) for feat_img in feat_imgs], dim=0).contiguous()
        
        
        ## Positonal embedding for query
        if self.with_bev_embedding:
            bev_queries = self.bev_embedding.weight.to(feat_imgs[0].dtype).view(self.nx[0], self.nx[1], c_img).permute(2, 0, 1).unsqueeze(0)
            query_pos = bev_queries.repeat(bs, 1, 1, 1).permute(0, 2, 3, 1).contiguous().view(-1, c_img)
        
        if self.positional_encoding is not None:
            query_pos_ = self.positional_encoding(self.pos_bev)
            query_pos_ = query_pos_.repeat(bs, 1, 1).permute(0, 2, 1).contiguous().view(-1, c_img)
        
        if self.with_bev_embedding and self.positional_encoding is not None:
            query_pos = query_pos + query_pos_
        elif self.positional_encoding is not None:
            query_pos = query_pos_

        
        # [bs, C, H, W] -> [bs, W, H, C]
        query = feat_pts.permute(0, 3, 2, 1).contiguous().view(-1, c_pts)
        if self.linear_layer is not None:
            query = self.linear_layer(query)
        
        
        for layer in self.depth_attn:
            output, attn_weights = layer(query=query, key=key, value=value, 
                                        ranks_feat_f=ranks_feat, ranks_bev_f=ranks_bev, 
                                        interval_starts_f=interval_starts, interval_lengths_f=interval_lengths,
                                        output_shape=output_shape, query_pos=query_pos)
        query = output
        
        # Reverse W and H
        # [bs, W, H, C] -> [bs, C, H, W]
        x = query.view(output_shape)
        x = x.permute(0, 3, 2, 1)
        
        if self.downsample is not None:
            x = self.downsample(x)
        

        weights = attn_weights.sum(dim=1) / attn_weights.shape[1]
        
        if sparse_bev_ind is not None:
            kept = torch.ones(
                ranks_feat.shape[0], device=ranks_feat.device, dtype=torch.bool)
            kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
            
            bev_idx = (~kept).to(torch.int32)
            offset = torch.cat([torch.tensor([1]).to(interval_lengths.device), interval_lengths[:-1]], dim=0).to(torch.int32)
            bev_idx[bev_idx == 0] = -offset+1
            bev_idx = bev_idx.cumsum(0)
            
            sparse_bev_ind = sparse_bev_ind.permute(0, 2, 1).contiguous()
            sparse_bev_ind = sparse_bev_ind.view(-1).to(torch.bool)
            gt_ind = sparse_bev_ind[ranks_bev.to(torch.int64)] & sparse_per_ind[ranks_feat.to(torch.int64)].to(torch.bool)
            
            per_idx = ranks_feat[gt_ind]
            bev_idx = bev_idx[gt_ind]
            
            per_idx, indices = torch.unique(per_idx, return_inverse=True)
            bev_idx = bev_idx[torch.unique(indices)]
            

            loss_cl = self.get_cl_loss(per_idx, bev_idx, weights)
        else:
            loss_cl = None
        
        return x, loss_cl
        # return x
    

    # def get_downsampled_gt_depth(self, gt_depths, downsample, depth_bins):
    #     """ Get ground-truth depth with projected Lidar points.

    #     Args:
    #         gt_depths (torch.tensor): _description_
    #         downsample (int): Downsample factor defined by img_size/feat_size
    #         depth_bins (int): Discreted depth number

    #     Returns:
    #         _type_: _description_
    #     """
    #     # downsample = int(self.image_size[0] / self.feature_size[0])
        
    #     B, N, H, W = gt_depths.shape
    #     gt_depths = gt_depths.view(B * N, H // downsample,
    #                                downsample, W // downsample,
    #                                downsample, 1)
    #     gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
    #     gt_depths = gt_depths.view(-1, downsample * downsample)
    #     gt_depths_tmp = torch.where(gt_depths == 0.0,
    #                                 1e5 * torch.ones_like(gt_depths),
    #                                 gt_depths)
    #     gt_depths = torch.min(gt_depths_tmp, dim=-1).values
    #     gt_depths = gt_depths.view(B * N, H // downsample,
    #                                W // downsample)

    #     gt_depths = (gt_depths - (self.dbound[0]-self.dbound[2])) / \
    #                     self.dbound[2]

    #     gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
    #                             gt_depths, torch.zeros_like(gt_depths))
    #     gt_depths = F.one_hot(
    #         gt_depths.long(), num_classes=depth_bins + 1).view(-1, depth_bins + 1)[:,
    #                                                                        1:]
    #     return gt_depths.float()
        
    def get_cl_loss(self, per_idx, bev_idx, weights):
        
        if per_idx.shape[0] == 0:
            return None
        
        valid_weights = weights[per_idx.to(torch.int64), :]
        valid_weights = torch.clamp(valid_weights, min=0.0000001)
        num_classes = weights.shape[1]
        assert bev_idx.max() < num_classes

        # 转换为 one-hot 向量
        # target = F.one_hot(bev_idx, num_classes=num_classes)

        with autocast(enabled=False):
            loss_cl = self.criterion(
                valid_weights,
                bev_idx,
            )
            
        return self.loss_cl_weight * loss_cl