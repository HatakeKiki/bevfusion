from mmcv.cnn.bricks.registry import ATTENTION
import torch
import torch.nn as nn
from mmdet3d.ops.depth_attn import depth_attn_weight, depth_attn_output
from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmcv.cnn import build_norm_layer
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_


@ATTENTION.register_module()
class DepthAttnLayer(nn.Module):
    
    def __init__(self,
                 embed_dim=256, 
                 num_heads=8, 
                 depth_attn=True, 
                 sem_attn=False,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=512,
                     num_fcs=2,
                     ffn_drop=0.1,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg=dict(type='LN'),
                 dropout_p=0.1,
                 ):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p
        
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        out_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj_weight = out_proj.weight
        self.out_proj_bias = out_proj.bias
        
        self.depth_attn = depth_attn
        self.sem_attn = sem_attn
        
        if self.depth_attn and self.sem_attn:
            self.in_proj_weight = Parameter(torch.empty(4 * embed_dim, embed_dim))
            self.in_proj_bias = Parameter(torch.empty(4 * embed_dim))
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        

        self.ffn = build_feedforward_network(ffn_cfgs)
        self.norm1 = build_norm_layer(norm_cfg, self.embed_dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, self.embed_dim)[1]
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        
        self._reset_parameters()


    def _reset_parameters(self):
        
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        
        xavier_uniform_(self.out_proj_weight)
        constant_(self.out_proj_bias, 0.)
        
        
    def forward(self, query_depth, key, value, 
                ranks_feat_f, ranks_bev_f, 
                interval_starts_f, interval_lengths_f, 
                output_shape, query_sem=None, query_pos=None, key_pos=None):
        """_summary_

        Args:
            feat_img (Tensor): image feature map, with shape of 
                                [bs, num_cams, channel, h, w], [32, 88] in this case
            feat_pts (Tensor): lidar feature map, with shape of
                                [bs, channel, H, W], [180, 180] in this case
            bev_pos (Tensor): position of bev grid center, with shape of (2, H, W)
            index (Tensor): index for each grid obtained from LSS, with the shape of (bs*H*W, max_pts)
        """
        output_2 = self.multi_head_attn(query_depth=query_depth, key=key, value=value, query_sem=query_sem,
                                        head_num=self.num_heads, ranks_feat_f=ranks_feat_f, ranks_bev_f=ranks_bev_f, 
                                        interval_starts_f=interval_starts_f, interval_lengths_f=interval_lengths_f,
                                        query_pos=query_pos, key_pos=key_pos, output_shape=output_shape, training=self.training, dropout=self.dropout_p)
        ## Add & Norm
        output = query_depth + self.dropout1(output_2)
        output = self.norm1(output)
        ## FFN: linear(dropout(activation(linear(output))))
        output_2 = self.ffn(output)
        ## Add & norm
        output = output + self.dropout2(output_2)
        output = self.norm2(output)

        return output_2
        
    def multi_head_attn(self, query_depth, key, value,
                        head_num, ranks_feat_f, ranks_bev_f, 
                        interval_lengths_f, interval_starts_f,
                        output_shape, query_sem=None, query_pos=None,
                        key_pos=None, dropout=0.1, training=False):
        """_summary_

        Args:
            query (Tensor): [bs*H*W, channel]
            key (_type_): [bs*h*w*num_cams, channel]
            value (_type_): [ba*h*w*num_cams, channel]
            dropout (float): _description_
        """

        assert query_depth is not None
        assert torch.equal(key, value)
            
        if query_pos is not None:
            query_depth = query_depth + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        head_dim = int(self.embed_dim / self.num_heads)
        scaling = float(head_dim) ** -0.5
        tgt_len, _ = query_depth.size()


        _start = 0
        _end = self.embed_dim*2
        _w = self.in_proj_weight[_start:_end, :]
        _b = self.in_proj_bias[_start:_end]
        
        k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        _start = self.embed_dim*2
        _end = self.embed_dim*3
        _w = self.in_proj_weight[_start:_end, :]
        _b = self.in_proj_bias[_start:_end]
            
        q_depth = F.linear(query_depth, _w, _b)
        q_depth = q_depth * scaling
        
        if query_sem is not None and self.sem_attn:
            
            _start = self.embed_dim*3
            _w = self.in_proj_weight[_start:, :]
            _b = self.in_proj_bias[_start:]
                
            q_sem = F.linear(query_sem, _w, _b)
            q_sem = q_sem * scaling
            q_depth = q_depth + q_sem
            
            assert False
        

        ref_num_f = int(interval_lengths_f.max())
        output_weight_shape = (tgt_len, head_num, ref_num_f)

        ## Q*K^T
        ranks_ref, attn_output_weights = depth_attn_weight(q_depth, k, 
                                                            ranks_feat_f, ranks_bev_f, 
                                                            interval_starts_f, interval_lengths_f,
                                                            output_weight_shape)
        ## softmax(weights)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=dropout, training=training)
        # attn_output_weights = attn_output_weights.new_ones(attn_output_weights.shape)
        
        ## weights * V
        ## [bs, W, H, C]
        attn_output = depth_attn_output(attn_output_weights, value, 
                                        ranks_feat_f, ranks_bev_f, 
                                        interval_starts_f, interval_lengths_f, 
                                        ranks_ref, output_shape)
        
        # ## mean instead of sum
        # order = ranks_bev_f.argsort()
        # ranks_bev_b = ranks_bev_f[order]
        # kept = torch.ones(
        #     ranks_bev_b.shape[0], device=ranks_bev_b.device, dtype=torch.bool)
        # kept[1:] = ranks_bev_b[1:] != ranks_bev_b[:-1]
        # interval_starts_b = torch.where(kept)[0].int()
        # interval_lengths_b = torch.zeros_like(interval_starts_b)
        # interval_lengths_b[:-1] = interval_starts_b[1:] - interval_starts_b[:-1]
        # interval_lengths_b[-1] = ranks_bev_b.shape[0] - interval_starts_b[-1]
        
        # # output_1 = attn_output_weights_1
        # pts_num = torch.ones(tgt_len).to(device=query_depth.device)
        
        # kept = torch.ones(
        #     ranks_bev_b.shape[0], device=ranks_bev_b.device, dtype=torch.bool)
        # kept[1:] = ranks_bev_b[1:] != ranks_bev_b[:-1]
        
        # index = ranks_bev_b[kept].to(torch.int64)
        # src = interval_lengths_b.to(pts_num.dtype)
        # pts_num = pts_num.scatter(index=index, src=src, dim=0)
        # pts_num = pts_num.view(output_shape[:-1]).unsqueeze(3)
        
        # attn_output = attn_output / pts_num
        
        
        # bs, W, H, C = attn_output.shape
        attn_output = attn_output.view(tgt_len, -1)
        attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
        
        return attn_output