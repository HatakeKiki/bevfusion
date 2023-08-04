from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
# for TTA
from mmdet3d.core.bbox import box_np_ops as box_np_ops

from mmdet3d.models.builder import (
    build_backbone,
    build_neck,
)
from mmdet3d.ops import Voxelization
from mmdet3d.models import FUSIONMODELS
from mmdet3d.ops.furthest_point_sample.furthest_point_sample import FurthestPointSampling, FurthestPointSamplingWithDist
from mmdet3d.ops.furthest_point_sample.points_sampler import Points_Sampler

from .base import Base3DFusionModel

__all__ = ["SimBEV"]


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def forward(self, x, label, topk=(1, 5)):
        x = x.squeeze()
        loss = self.criterion(x, label)
        acc = self.accuracy(x, label, topk=topk)
        return loss, acc
    
@FUSIONMODELS.register_module()
class SimBEV(Base3DFusionModel):
    def __init__(
        self,
        img_backbone: Dict[str, Any],
        img_neck: Dict[str, Any],
        pts_voxelize: Dict[str, Any],
        pts_backbone: Dict[str, Any],
        pts_mlp: Dict[str, Any],
        img_mlp: Dict[str, Any],
        pts_sampler: Dict[str, Any],
        T: float = 0.07,
        **kwargs,
    ) -> None:
        super().__init__()

        self.img_backbone = build_backbone(img_backbone),
        self.img_neck = build_neck(img_neck),
        self.img_backbone = self.img_backbone[0]
        self.img_neck = self.img_neck[0]

        self.pts_size = pts_voxelize['point_cloud_range'][3:5]
        self.pts_voxelize = Voxelization(**pts_voxelize)
        self.pts_backbone = build_backbone(pts_backbone)

        self.pts_mlp = projection_MLP(in_dim=pts_mlp['in_dim'], 
                                      hidden_dim=pts_mlp['hidden_dim'],
                                      out_dim=pts_mlp['out_dim'])
        self.img_mlp = projection_MLP(in_dim=img_mlp['in_dim'], 
                                      hidden_dim=img_mlp['hidden_dim'],
                                      out_dim=img_mlp['out_dim'])
        self.loss = NCESoftmaxLoss()
        # self.pts_sampler = FurthestPointSampling()
        self.pts_sampler = Points_Sampler(**pts_sampler)
        self.pts_num = pts_sampler['num_point'][0]
        self.T = T
    
    def init_weights(self) -> None:
        self.img_backbone.init_weights()
        return
            
    def extract_camera_features(self,img) -> torch.Tensor:
        B, N, C, H, W = img.size()
        img = img.view(B*N, C, H, W)
        x = self.img_backbone(img)
        x = self.img_neck(x)
        
        if not isinstance(x, torch.Tensor):
            x = x[0]
            
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.pts_backbone(feats, coords, batch_size, sizes=sizes)
        return x
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxelize(res)
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
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
            
        feats = feats.contiguous()
        
        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points", "points_single"))
    def forward(
        self,
        points,
        points_single,
        img,
        metas,
        lidar2image=None,
        lidar_aug_matrix=None,
        img_aug_matrix=None,
        **kwargs,
    ):
        
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                        points,
                        points_single,
                        img,
                        metas,
                        lidar2image=lidar2image,
                        lidar_aug_matrix=lidar_aug_matrix,
                        img_aug_matrix=img_aug_matrix,
                        **kwargs,
            )
            return outputs
        
    def positive_sample(self,
                        pts_feat,
                        img_feat,
                        points,
                        lidar2image,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        img_size,
                        pts_size):
        
        B, N, C, fh, fw = img_feat.shape
        # img_feat = img_feat.view(B*N, C, fh, fw)
        # B = img_feat.shape[0]
        
        # pts_grids = []
        # img_grids = []
        img_feat_sampled_list = []
        pts_feat_sampled_list = []
        for b in range(B):
            ori_coords = points[b][:, :2]
            ori_coords = torch.cat([ori_coords, ori_coords.new_zeros((ori_coords.shape[0], 1))], dim=-1)
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
            # # get 2d coords
            # dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e4)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < img_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < img_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            
            on_img_pts = on_img.sum(dim=0).to(bool)
            ori_coords = ori_coords[on_img_pts]
            cur_coords = cur_coords[:, on_img_pts, :]
            on_img = on_img[:, on_img_pts]
            
            
            indices = self.pts_sampler(ori_coords.unsqueeze(0), cur_coords.unsqueeze(0))[0]
            
            ori_coords = ori_coords[indices.to(torch.int64), :]
            cur_coords = cur_coords[:, indices.to(torch.int64), :]
            on_img = on_img[:, indices.to(torch.int64)]
            
            assert (on_img.sum(dim=0) > 2).sum() == 0
            
            hits_two_locs = (on_img.sum(dim=0) == 2)
            hits_two = on_img[:, hits_two_locs].transpose(0, 1)
            if hits_two_locs.sum() > 0:
                hits_two_indices = torch.nonzero(hits_two)
                rand_choice = torch.rand(int(hits_two_indices.shape[0]/2)) > 0.5
                
                rand_choice = (torch.range(0, int(hits_two_indices.shape[0]/2) - 1) * 2 + rand_choice).to(torch.int64)
                # rand_choice = torch.stack([torch.range(0, hits_two_indices.shape[0]-1), rand_choice], dim=1).to(torch.int64)
                hits_two_indices = hits_two_indices[rand_choice, :]
                
                _, n_views = hits_two.shape
                
                hits_two_indices = hits_two_indices[:, 0] * n_views +  hits_two_indices[:, 1]
                
                hits_two = hits_two.reshape(-1)
                hits_two[hits_two_indices] = False
                hits_two = hits_two.reshape(-1, n_views)
                
                on_img[:, hits_two_locs] = hits_two.transpose(0, 1)
                
            assert on_img.sum() == self.pts_num

            
            pts_grids = []
            for c in range(N):
                masked_cur_coords = cur_coords[c, on_img[c]]
                masked_ori_coords = ori_coords[on_img[c]]
                
                h, w = img_size
                H, W = pts_size
                
                coor_v, coor_u = torch.split(masked_cur_coords, 1, dim=1)  # each is Nx1
                coor_u = coor_u / w * 2 - 1
                coor_v = coor_v / h * 2 - 1
                
                coor_x, coor_y, _ = torch.split(masked_ori_coords, 1, dim=1)  # each is Nx1
                coor_x = (coor_x + W) / W - 1
                coor_y = (coor_y + H) / H - 1
                
                img_grid = torch.cat([coor_u, coor_v],
                                dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2
                pts_grid = torch.cat([coor_x, coor_y],
                                dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2
                
                pts_grids.append(pts_grid)
                # img_grids.append(img_grid)
                
                img_feat_sampled = F.grid_sample(img_feat[b, c, ...].unsqueeze(0),
                                                img_grid,
                                                mode='bilinear',
                                                padding_mode='zeros',
                                                align_corners=True)
                img_feat_sampled_list.append(img_feat_sampled)
                
            pts_grids = torch.cat(pts_grids, dim=-2)
            pts_feat_sampled = F.grid_sample(pts_feat[b, ...].unsqueeze(0),
                                            pts_grids,
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=True)
            pts_feat_sampled_list.append(pts_feat_sampled)
            

        pts_feat_sampled_list = torch.cat(pts_feat_sampled_list, dim=-1).squeeze()
        img_feat_sampled_list = torch.cat(img_feat_sampled_list, dim=-1).squeeze()
        return pts_feat_sampled_list, img_feat_sampled_list
    
    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        points,
        points_single,
        img,
        metas,
        lidar2image,
        lidar_aug_matrix,
        img_aug_matrix,
        **kwargs,
    ):
        img_feat = self.extract_camera_features(img)
        pts_feat = self.extract_lidar_features(points)
        img_size = img.shape[-2:]
        # feat_size = img_feat.shape[-2:]
        pts_feat_sampled, img_feat_sampled = self.positive_sample(pts_feat,
                                                                img_feat,
                                                                points_single,
                                                                lidar2image,
                                                                img_aug_matrix,
                                                                lidar_aug_matrix,
                                                                img_size,
                                                                self.pts_size)
        
        pts_feat_sampled = pts_feat_sampled.transpose(0, 1)
        img_feat_sampled = img_feat_sampled.transpose(0, 1)
        pts_feat_sampled_norm = F.normalize(self.pts_mlp(pts_feat_sampled), dim=1)
        img_feat_sampled_norm = F.normalize(self.img_mlp(img_feat_sampled), dim=1)
        
        N, _ =  pts_feat_sampled.shape
        labels = torch.arange(N).cuda().long()
        
        logits = torch.mm(pts_feat_sampled_norm, img_feat_sampled_norm.transpose(0, 1))
        out = torch.div(logits, self.T)
        loss_0, acc_0 = self.loss(out, labels, topk=(int(N / 10), int(N / 2)))
        loss_1, acc_1 = self.loss(out.T, labels, topk=(int(N / 10), int(N / 2)))
        
        loss = dict()
        loss['loss_0'] = loss_0
        loss['acc_0_top1'] = acc_0[0]
        loss['acc_0_top5'] = acc_0[1]
        
        loss['loss_1'] = loss_1
        loss['acc_1_top1'] = acc_1[0]
        loss['acc_1_top5'] = acc_1[1]
        
        return loss
        
        

