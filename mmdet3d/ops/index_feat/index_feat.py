# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

from . import index_feat_ext

__all__ = ['index_feat']


class QuickIndexFeat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feat, 
                ranks_feat, ranks_bev, 
                index_feat_shape, 
                interval_starts, interval_lengths):
        
        feat = feat.contiguous().float()
        
        ranks_bev = ranks_bev.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        attn_mask = feat.new_ones(ranks_feat.shape).int() * (-1)
        out = feat.new_zeros(index_feat_shape)

        index_feat_ext.index_feat_forward(
            feat,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
            attn_mask,
            out,
        )

        ctx.save_for_backward(ranks_bev, ranks_feat, feat, attn_mask)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, ranks_feat, feat, attn_mask = ctx.saved_tensors
        
        # ranks by camera feature
        order = ranks_feat.argsort()
        ranks_feat, ranks_bev, attn_mask = \
            ranks_feat[order], ranks_bev[order], attn_mask[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]


        feat = feat.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()
        
        out_grad = out_grad.contiguous()
        attn_mask = attn_mask.contiguous()
        feat_grad = feat.new_zeros(feat.shape)

        index_feat_ext.index_feat_backward(
            out_grad,
            feat_grad,
            feat,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
            attn_mask,
        )
        return feat_grad, None, None, None, None, None


def index_feat(feat, ranks_feat, ranks_bev, index_feat_shape, 
               interval_starts, interval_lengths):
    x = QuickIndexFeat.apply(feat, ranks_feat, ranks_bev,
                              index_feat_shape, interval_starts,
                              interval_lengths)
    # x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x.contiguous()
