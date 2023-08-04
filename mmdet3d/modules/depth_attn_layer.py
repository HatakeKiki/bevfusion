from mmcv.cnn.bricks.registry import ATTENTION
import torch
import torch.nn as nn
from mmdet3d.ops.depth_attn import depth_attn_weight, depth_attn_output
from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmcv.cnn import build_norm_layer
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_


@ATTENTION.register_module()
class DepthAttnLayer(nn.Module):
    
    def __init__(self,
                 with_res=False,
                 embed_dim=256, 
                 num_heads=8, 
                 act_cfg=dict(type='ReLU', inplace=True),
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
        self.with_res = with_res
        
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # out_proj = Linear(self.embed_dim, 2*self.embed_dim, bias=True)
        # self.out_proj_weight = out_proj.weight
        # self.out_proj_bias = out_proj.bias
        
        self.in_proj_q = nn.Sequential(
            nn.Linear(self.embed_dim, 2*self.embed_dim),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(2*self.embed_dim, self.embed_dim),
        )
        
        self.in_proj_k = nn.Sequential(
            nn.Linear(self.embed_dim, 2*self.embed_dim),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(2*self.embed_dim, self.embed_dim),
        )
        self.norm = build_norm_layer(norm_cfg, self.embed_dim)[1]
        
        if self.with_res:
            # self.ffn = build_feedforward_network(ffn_cfgs)
            self.norm1 = build_norm_layer(norm_cfg, self.embed_dim)[1]
            # self.norm2 = build_norm_layer(norm_cfg, self.embed_dim)[1]
            self.dropout1 = nn.Dropout(dropout_p)
            # self.dropout2 = nn.Dropout(dropout_p)
        
    #     self._reset_parameters()


    # def _reset_parameters(self):
        
    #     xavier_uniform_(self.out_proj_weight)
    #     constant_(self.out_proj_bias, 0.)
        
        
    def forward(self, query, key, value, 
                ranks_feat_f, ranks_bev_f, 
                interval_starts_f, interval_lengths_f, 
                output_shape, query_pos=None, key_pos=None):
        """_summary_

        Args:
            feat_img (Tensor): image feature map, with shape of 
                                [bs, num_cams, channel, h, w], [32, 88] in this case
            feat_pts (Tensor): lidar feature map, with shape of
                                [bs, channel, H, W], [180, 180] in this case
            bev_pos (Tensor): position of bev grid center, with shape of (2, H, W)
            index (Tensor): index for each grid obtained from LSS, with the shape of (bs*H*W, max_pts)
        """
        output_2, attn_weights = self.multi_head_attn(query=query, key=key, value=value,
                                        head_num=self.num_heads, ranks_feat_f=ranks_feat_f, ranks_bev_f=ranks_bev_f, 
                                        interval_starts_f=interval_starts_f, interval_lengths_f=interval_lengths_f,
                                        query_pos=query_pos, key_pos=key_pos, output_shape=output_shape, 
                                        training=self.training, dropout=self.dropout_p, keep_weight=True)
        if self.with_res:
            ## Add & Norm
            output = query + self.dropout1(output_2)
            output_2 = self.norm1(output)
            ## FFN: linear(dropout(activation(linear(output))))
            # output_2 = self.ffn(output)
            # ## Add & norm
            # output = output + self.dropout2(output_2)
            # output_2 = self.norm2(output)

        return output_2, attn_weights
        
    def multi_head_attn(self, query, key, value,
                        head_num, ranks_feat_f, ranks_bev_f, 
                        interval_lengths_f, interval_starts_f,
                        output_shape, query_pos=None,
                        key_pos=None, dropout=0.1, training=False, keep_weight=False):
        """_summary_

        Args:
            query (Tensor): [bs*H*W, channel]
            key (_type_): [bs*h*w*num_cams, channel]
            value (_type_): [ba*h*w*num_cams, channel]
            dropout (float): _description_
        """
        # assert torch.equal(key, value)
            
        if key_pos is not None:
            key = key + key_pos
        
        head_dim = int(self.embed_dim / self.num_heads)
        scaling = float(head_dim) ** -0.5
        tgt_len, _ = query.size()
        k_len, _ = key.size()
        
        ## Unlinear projection for QKV
        if query_pos is not None:
            q = self.in_proj_q(query+query_pos)
        else:
            q = self.in_proj_q(query)

        q = q * scaling
        k = self.in_proj_k(key)
        
        # q = F.normalize(q, dim=1)
        # k = F.normalize(k, dim=1)

        ref_num_f = int(interval_lengths_f.max())
        output_weight_shape = (k_len, head_num, ref_num_f)

        ## Q*K^T
        ranks_ref, attn_output_weights = depth_attn_weight(q, k, 
                                                        ranks_feat_f, ranks_bev_f, 
                                                        interval_starts_f, interval_lengths_f,
                                                        output_weight_shape)
        ## softmax(weights)
        attn_output_weights_sm = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=dropout, training=training)

        ## weights * V
        ## [bs, W, H, C]
        attn_output = depth_attn_output(attn_output_weights_sm, value, 
                                        ranks_feat_f, ranks_bev_f, 
                                        interval_starts_f, interval_lengths_f, 
                                        ranks_ref, output_shape)
        attn_output = attn_output.view(tgt_len, -1)
        attn_output = F.dropout(attn_output, p=dropout, training=training)
        # attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
        
        attn_output = self.norm(attn_output)
        
        if keep_weight:
            return attn_output, attn_output_weights_sm
        else:
            return attn_output