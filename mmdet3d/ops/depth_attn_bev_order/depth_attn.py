# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

from . import depth_attn_ext

__all__ = ['depth_attn_weight', 'depth_attn_output']

class QuickDepthAttnWeight(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, feat_bev, feat_k,
                ranks_feat, ranks_bev,
                interval_starts, interval_lengths,
                output_weight_shape):
        """_summary_

        Args:
            ctx (_type_): _description_
            feat_bev (_type_): [bs*H*W, c]
            feat_k (_type_): [bs*N*h*w, c]
            ranks_feat (_type_): _description_
            ranks_bev (_type_): _description_
            interval_starts (_type_): _description_
            interval_lengths (_type_): _description_
            output_weight_shape (_type_): (bs*H*W, head_num, ref_num)

        Returns:
            _type_: _description_
        """
        feat_bev = feat_bev.contiguous().float()
        feat_k = feat_k.contiguous().float()
                
        ranks_feat = ranks_feat.contiguous().int()
        ranks_bev = ranks_bev.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()
        
        key_padding_mask = feat_bev.new_ones(ranks_bev.shape).int() * int(-1)
        q_padding_mask = feat_bev.new_ones(feat_bev.shape[0]).bool()
        # out_weight = feat_bev.new_zeros(output_weight_shape)
        out_weight = feat_bev.new_ones(output_weight_shape) * float('-inf')
        depth_attn_ext.depth_attn_weight_forward(feat_bev, feat_k, 
                                                 ranks_feat, ranks_bev, 
                                                 interval_lengths, interval_starts, 
                                                 key_padding_mask, q_padding_mask, out_weight)
        # print(interval_lengths.shape)
        ctx.save_for_backward(feat_bev, feat_k, 
                              ranks_feat, ranks_bev, 
                              interval_starts, interval_lengths, 
                              key_padding_mask)
        
        return key_padding_mask, q_padding_mask, out_weight
    
    def backward(ctx, key_padding_mask_grad, q_padding_mask_grad, out_grad):
        """_summary_

        Args:
            ctx (_type_): _description_
            out_grad (_type_): [bs*H*W, head_num_ref_num]

        Returns:
            _type_: _description_
        """
        
        (feat_bev, feat_k, ranks_feat, ranks_bev, 
            interval_starts, interval_lengths, key_padding_mask) = ctx.saved_tensors
        
        order = ranks_feat.argsort()
        ranks_feat_f, ranks_bev_f, key_padding_mask_f = \
            ranks_feat[order], ranks_bev[order], key_padding_mask[order]
        kept = torch.ones(
            ranks_bev_f.shape[0], device=ranks_bev_f.device, dtype=torch.bool)
        kept[1:] = ranks_feat_f[1:] != ranks_feat_f[:-1]
        interval_starts_f = torch.where(kept)[0].int()
        interval_lengths_f = torch.zeros_like(interval_starts_f)
        interval_lengths_f[:-1] = interval_starts_f[1:] - interval_starts_f[:-1]
        interval_lengths_f[-1] = ranks_bev_f.shape[0] - interval_starts_f[-1]
        
        feat_bev = feat_bev.contiguous()
        feat_k = feat_k.contiguous()

        ranks_feat_f = ranks_feat_f.contiguous()
        ranks_bev_f = ranks_bev_f.contiguous()
        interval_lengths_f = interval_lengths_f.contiguous()
        interval_starts_f = interval_starts_f.contiguous()
        
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths = interval_lengths.contiguous()
        interval_starts = interval_starts.contiguous()
        key_padding_mask_f = key_padding_mask_f.contiguous()
        
        out_grad = out_grad.contiguous()
        
        bev_grad = feat_bev.new_zeros(feat_bev.shape)
        k_grad = feat_bev.new_zeros(feat_k.shape)
        
        
        
        
        depth_attn_ext.depth_attn_weight_backward(out_grad, bev_grad, k_grad,
                                                  feat_bev, feat_k, 
                                                  ranks_feat, ranks_bev,
                                                  ranks_feat_f, ranks_bev_f,
                                                  interval_lengths, interval_starts,
                                                  interval_lengths_f, interval_starts_f,
                                                  key_padding_mask_f)
        
        return bev_grad, k_grad, None, None, None, None, None

class QuickDepthAttnOutput(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, weight, feat_v,
                ranks_feat, ranks_bev,
                interval_starts, interval_lengths,
                key_padding_mask, output_shape):
        """_summary_

        Args:
            ctx (_type_): _description_
            weight (_type_): [bs*H*W, head_num, ref_num]
            feat_v (_type_): [bs*N*h*w, c]
            ranks_feat (_type_): _description_
            ranks_bev (_type_): _description_
            interval_starts (_type_): _description_
            interval_lengths (_type_): _description_
            key_padding_mask (_type_): _description_
            output_shape (_type_): [bs, H, W, c]

        Returns:
            _type_: _description_
        """
        
        weight = weight.contiguous().float()
        feat_v = feat_v.contiguous().float()
                
        ranks_feat = ranks_feat.contiguous().int()
        ranks_bev = ranks_bev.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()
        
        out = feat_v.new_zeros(output_shape)
        depth_attn_ext.depth_attn_output_forward(weight, feat_v, 
                                                 ranks_feat, ranks_bev, 
                                                 interval_lengths, interval_starts, 
                                                 out)
        
        ctx.save_for_backward(weight, feat_v, 
                              ranks_feat, ranks_bev, 
                              interval_starts, interval_lengths, 
                              key_padding_mask)
        
        return out
    
    def backward(ctx, out_grad):
        
        (weight, feat_v, ranks_feat, ranks_bev, 
            interval_starts, interval_lengths, key_padding_mask) = ctx.saved_tensors
        
        order = ranks_feat.argsort()
        ranks_feat_f, ranks_bev_f, key_padding_mask_f = \
            ranks_feat[order], ranks_bev[order], key_padding_mask[order]
        kept = torch.ones(
            ranks_bev_f.shape[0], device=ranks_bev_f.device, dtype=torch.bool)
        kept[1:] = ranks_feat_f[1:] != ranks_feat_f[:-1]
        interval_starts_f = torch.where(kept)[0].int()
        interval_lengths_f = torch.zeros_like(interval_starts_f)
        interval_lengths_f[:-1] = interval_starts_f[
            1:] - interval_starts_f[:-1]
        interval_lengths_f[-1] = ranks_bev_f.shape[0] - interval_starts_f[-1]
        
        weight = weight.contiguous()
        feat_v = feat_v.contiguous()

        ranks_feat_f = ranks_feat_f.contiguous()
        ranks_bev_f = ranks_bev_f.contiguous()
        interval_lengths_f = interval_lengths_f.contiguous()
        interval_starts_f = interval_starts_f.contiguous()
        
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths = interval_lengths.contiguous()
        interval_starts = interval_starts.contiguous()
        key_padding_mask_f = key_padding_mask_f.contiguous()
        
        out_grad = out_grad.contiguous()
        
        w_grad = weight.new_zeros(weight.shape)
        v_grad = feat_v.new_zeros(feat_v.shape)
        
        depth_attn_ext.depth_attn_output_backward(out_grad, w_grad, v_grad,
                                                  weight, feat_v, 
                                                  ranks_feat, ranks_bev,
                                                  ranks_feat_f, ranks_bev_f,
                                                  interval_lengths, interval_starts,
                                                  interval_lengths_f, interval_starts_f,
                                                  key_padding_mask_f)
        
        return w_grad, v_grad, None, None, None, None, None, None
        
        
        

def depth_attn_weight(feat_bev, feat_k, 
                      ranks_feat, ranks_bev, 
                      interval_starts, interval_lengths,
                      output_weight_shape):
    
    out = QuickDepthAttnWeight.apply(
        feat_bev, feat_k, 
        ranks_feat, ranks_bev, 
        interval_starts, interval_lengths,
        output_weight_shape
    )
    
    return out

def depth_attn_output(weight, feat_v,
                        ranks_feat, ranks_bev,
                        interval_starts, interval_lengths,
                        key_padding_mask, output_shape):
    
    x = QuickDepthAttnOutput.apply(weight, feat_v,
                                    ranks_feat, ranks_bev,
                                    interval_starts, interval_lengths,
                                    key_padding_mask, output_shape)
    return x



def test_depth_attn():

    return
