import torch
from torch import nn
from mmdet3d.models.backbones.dsvt_input_layer import DSVTInputLayer
from mmdet3d.models.backbones.dsvt import Stage_ReductionAtt_Block
from mmdet3d.models.builder import FUSERS
from mmdet3d.models.backbones.pointpillar3d_scatter import PointPillarScatter3d
import copy
from mmcv.cnn import build_norm_layer

__all__ = ["SparsePool"]


@FUSERS.register_module()
class SparsePool(nn.Module):
    def __init__(self, input_layers: dict, map_to_bev_module: dict, d_model: list, pool_volume: int):
        super().__init__()
        self.d_model = d_model
        self.pool_volume = pool_volume
        self.input_layer = DSVTInputLayer(input_layers)
        self.reduction = Stage_ReductionAtt_Block(d_model[0], pool_volume)

        self.pillar_scater = PointPillarScatter3d(input_shape=[360, 360, 1], num_bev_features=128)
        
        self.norm = build_norm_layer(dict(type='LN'), 256)[1]
        

    def forward(self, batch_dict: dict):
        
        coords = batch_dict['voxel_coords']
        voxel_feat_img = batch_dict['spatial_features_img'][coords[:, 0], :, coords[:, 2], coords[:, 3]]
        voxel_feat = batch_dict['voxel_features']
        voxel_num, voxel_channel = voxel_feat.shape
        
        voxel_feat_img = self.norm(voxel_feat_img)
        
        voxel_feat_img = voxel_feat_img.view(voxel_num, voxel_channel, -1).permute(0, 2, 1).contiguous()
        voxel_feat_img = voxel_feat_img.view(-1, voxel_channel)
        coords_1 = copy.deepcopy(coords)
        coords_1[:, 1] = 1
        coords_2 = copy.deepcopy(coords)
        coords_2[:, 1] = 2
        coords_voxel = torch.cat([coords, coords_1, coords_2], dim=0)
        voxel_feat = torch.cat([voxel_feat, voxel_feat_img], dim=0)
        
        batch_dict['voxel_coords'] = coords_voxel
        batch_dict['voxel_features'] = voxel_feat
        
        voxel_info = self.input_layer(batch_dict)
        
        voxel_feat = voxel_info['voxel_feats_stage0']
        pooling_mapping_index = voxel_info['pooling_mapping_index_stage1']
        pooling_index_in_pool = voxel_info['pooling_index_in_pool_stage1']
        pooling_preholder_feats = voxel_info['pooling_preholder_feats_stage1']
        
    
        prepool_features = pooling_preholder_feats.type_as(voxel_feat)
        pooled_voxel_num = prepool_features.shape[0]
        pool_volume = prepool_features.shape[1]
        prepool_features[pooling_mapping_index, pooling_index_in_pool] = voxel_feat
        # prepool_features = prepool_features.view(prepool_features.shape[0], -1)
        
        prepool_features = prepool_features.view(pooled_voxel_num, self.pool_volume, -1).permute(0, 2, 1)
        key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(prepool_features.device).int()
        
        output = self.reduction(prepool_features, key_padding_mask)
        
        batch_dict['voxel_features'] = output
        batch_dict['pillar_features'] = output
        batch_dict['voxel_coords'] = coords
        self.pillar_scater(batch_dict)
        
        return batch_dict
                    
        
