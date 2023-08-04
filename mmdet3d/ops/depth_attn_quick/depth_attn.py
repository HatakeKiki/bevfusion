import torch
from . import depth_attn_ext

__all__ = ['depth_attn_weight', 'depth_attn_output']

class QuickDepthAttnWeight(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, feat_bev, feat_k,
                ranks_feat_f, ranks_bev_f,
                interval_starts_f, interval_lengths_f,
                output_weight_shape):
        """_summary_

        Args:
            ctx (_type_): _description_
            feat_bev (_type_): [bs*H*W, c]
            feat_k (_type_): [bs*N*h*w, c]
            ranks_feat (_type_): in feat_order
            ranks_bev (_type_): _description_
            interval_starts (_type_): _description_
            interval_lengths (_type_): _description_
            output_weight_shape (_type_): [bs*N*h*w, head_num, ref_num]

        Returns:
            _type_: _description_
        """
        feat_bev = feat_bev.contiguous().float()
        feat_k = feat_k.contiguous().float()
                
        ranks_feat_f = ranks_feat_f.contiguous().int()
        ranks_bev_f = ranks_bev_f.contiguous().int()
        interval_lengths_f = interval_lengths_f.contiguous().int()
        interval_starts_f = interval_starts_f.contiguous().int()
        
        ranks_ref_f = feat_bev.new_ones(ranks_bev_f.shape).int() * int(-1)
        # out_weight = feat_bev.new_zeros(output_weight_shape)
        out_weight = feat_bev.new_ones(output_weight_shape) * float('-inf')
            
        depth_attn_ext.depth_attn_weight_forward(feat_bev, feat_k, 
                                                 ranks_feat_f, ranks_bev_f, 
                                                 interval_lengths_f, interval_starts_f, 
                                                 ranks_ref_f, out_weight)
        # print(interval_lengths.shape)
        ctx.save_for_backward(feat_bev, feat_k, 
                              ranks_feat_f, ranks_bev_f, 
                              interval_starts_f, interval_lengths_f, 
                              ranks_ref_f)
        
        return ranks_ref_f, out_weight
    
    def backward(ctx, ranks_ref_grad, out_grad):
        """_summary_

        Args:
            ctx (_type_): _description_
            out_grad (_type_): [bs*H*W, head_num_ref_num]

        Returns:
            _type_: _description_
        """
        
        (feat_bev, feat_k, ranks_feat_f, ranks_bev_f, 
            interval_starts_f, interval_lengths_f, ranks_ref_f) = ctx.saved_tensors
        
        order = ranks_bev_f.argsort()
        ranks_feat_b, ranks_bev_b, ranks_ref_b = \
            ranks_feat_f[order], ranks_bev_f[order], ranks_ref_f[order]
        kept = torch.ones(
            ranks_bev_b.shape[0], device=ranks_bev_b.device, dtype=torch.bool)
        kept[1:] = ranks_bev_b[1:] != ranks_bev_b[:-1]
        interval_starts_b = torch.where(kept)[0].int()
        interval_lengths_b = torch.zeros_like(interval_starts_b)
        interval_lengths_b[:-1] = interval_starts_b[1:] - interval_starts_b[:-1]
        interval_lengths_b[-1] = ranks_bev_b.shape[0] - interval_starts_b[-1]
        
        feat_bev = feat_bev.contiguous()
        feat_k = feat_k.contiguous()

        ranks_feat_f = ranks_feat_f.contiguous()
        ranks_bev_f = ranks_bev_f.contiguous()
        interval_lengths_f = interval_lengths_f.contiguous()
        interval_starts_f = interval_starts_f.contiguous()
        
        ranks_feat_b = ranks_feat_b.contiguous()
        ranks_bev_b = ranks_bev_b.contiguous()
        interval_lengths_b = interval_lengths_b.contiguous()
        interval_starts_b = interval_starts_b.contiguous()
        ranks_ref_b = ranks_ref_b.contiguous()
        
        out_grad = out_grad.contiguous()
        
        bev_grad = feat_bev.new_zeros(feat_bev.shape)
        k_grad = feat_bev.new_zeros(feat_k.shape)
        
        
        depth_attn_ext.depth_attn_weight_backward(out_grad, bev_grad, k_grad,
                                                  feat_bev, feat_k, 
                                                  ranks_feat_b, ranks_bev_b,
                                                  ranks_feat_f, ranks_bev_f, ranks_ref_b,
                                                  interval_lengths_b, interval_starts_b,
                                                  interval_lengths_f, interval_starts_f)
        
        return bev_grad, k_grad, None, None, None, None, None

class QuickDepthAttnOutput(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, weight, feat_v,
                ranks_feat_f, ranks_bev_f,
                interval_starts_f, interval_lengths_f,
                ranks_ref_f, output_shape):
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
        
        order = ranks_bev_f.argsort()
        ranks_feat_b, ranks_bev_b, ranks_ref_b = \
            ranks_feat_f[order], ranks_bev_f[order], ranks_ref_f[order]
        kept = torch.ones(
            ranks_bev_b.shape[0], device=ranks_bev_b.device, dtype=torch.bool)
        kept[1:] = ranks_bev_b[1:] != ranks_bev_b[:-1]
        interval_starts_b = torch.where(kept)[0].int()
        interval_lengths_b = torch.zeros_like(interval_starts_b)
        interval_lengths_b[:-1] = interval_starts_b[1:] - interval_starts_b[:-1]
        interval_lengths_b[-1] = ranks_bev_b.shape[0] - interval_starts_b[-1]
        
        
        ranks_feat_b = ranks_feat_b.contiguous().int()
        ranks_bev_b = ranks_bev_b.contiguous().int()
        interval_lengths_b = interval_lengths_b.contiguous().int()
        interval_starts_b = interval_starts_b.contiguous().int()
        
        ranks_feat_f = ranks_feat_f.contiguous().int()
        ranks_bev_f = ranks_bev_f.contiguous().int()
        interval_lengths_f = interval_lengths_f.contiguous().int()
        interval_starts_f = interval_starts_f.contiguous().int()
        
        out = feat_v.new_zeros(output_shape)
        depth_attn_ext.depth_attn_output_forward(weight, feat_v, 
                                                 ranks_feat_b, ranks_bev_b, ranks_ref_b,
                                                 interval_lengths_b, interval_starts_b, 
                                                 out)
        
        ctx.save_for_backward(weight, feat_v, 
                              ranks_feat_f, ranks_bev_f, ranks_ref_f,
                              interval_starts_f, interval_lengths_f)
        
        return out
    
    def backward(ctx, out_grad):
        
        (weight, feat_v, ranks_feat_f, ranks_bev_f, ranks_ref_f,
            interval_starts_f, interval_lengths_f) = ctx.saved_tensors

        
        weight = weight.contiguous()
        feat_v = feat_v.contiguous()

        ranks_feat_f = ranks_feat_f.contiguous()
        ranks_bev_f = ranks_bev_f.contiguous()
        interval_lengths_f = interval_lengths_f.contiguous()
        interval_starts_f = interval_starts_f.contiguous()
        
        out_grad = out_grad.contiguous()
        
        w_grad = weight.new_zeros(weight.shape)
        v_grad = feat_v.new_zeros(feat_v.shape)
        
        depth_attn_ext.depth_attn_output_backward(out_grad, w_grad, v_grad,
                                                  weight, feat_v, 
                                                  ranks_feat_f, ranks_bev_f, ranks_ref_f,
                                                  interval_lengths_f, interval_starts_f)
        
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
                        ranks_ref, output_shape):
    
    x = QuickDepthAttnOutput.apply(weight, feat_v,
                                    ranks_feat, ranks_bev,
                                    interval_starts, interval_lengths,
                                    ranks_ref, output_shape)
    return x



def test_depth_attn():

    return
