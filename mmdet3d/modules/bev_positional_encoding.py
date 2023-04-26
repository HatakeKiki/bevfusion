from mmcv.cnn.bricks.registry import POSITIONAL_ENCODING
from mmcv.runner import BaseModule
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@POSITIONAL_ENCODING.register_module()
class SineBEVPositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SineBEVPositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, pos_bev):
        """Forward function for `SinePositionalEncoding`.

        Args:
            pos_bev (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, H, W, 2].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        B, _, _, _ = pos_bev.size()
        # mask = mask.to(torch.int)
        # not_mask = 1 - mask  # logical_not
        # y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # x_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = pos_bev[:, 0, ...]
        y_embed = pos_bev[:, 1, ...]

        # pos_bev = pos_bev.to(torch.int)
        # not_mask = 1 - pos_bev  # logical_not
        # y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            x_embed = x_embed / (x_embed.max() + self.eps) * self.scale
            y_embed = y_embed / (y_embed.max() + self.eps) * self.scale
            
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=pos_bev.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str
    
@POSITIONAL_ENCODING.register_module()
class LearnedBEVPositionalEncoding(BaseModule):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=2, num_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_feats, kernel_size=1),
            nn.BatchNorm1d(num_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_feats, num_feats, kernel_size=1))

    def forward(self, pos):
        # pos = pos.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(pos)
        return position_embedding
    
@POSITIONAL_ENCODING.register_module()
class LearnedDepthPositionalEncoding(BaseModule):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=1, num_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_feats, kernel_size=1),
            nn.BatchNorm1d(num_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_feats, num_feats, kernel_size=1))

    def forward(self, pos):
        # pos = pos.transpose(1, 2).contiguous()
        pos = pos.transpose(1, 2).contiguous()
        dis = F.pairwise_distance(pos, torch.zeros_like(pos))
        dis = dis.unsqueeze(1)
        position_embedding = self.position_embedding_head(dis)
        return position_embedding