import torch.nn as nn

def fix_bn(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        if m.affine and m.weight.requires_grad is False:
            m.track_running_stats = False
            m.eval()