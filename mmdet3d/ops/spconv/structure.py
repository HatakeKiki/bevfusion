import numpy as np
import torch


def scatter_nd(indices, updates, shape, return_ind=False):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1] :]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    
    if return_ind:
        updates_indices = torch.ones((indices.shape[0], 1), dtype=torch.bool, device=updates.device)
        ret_indices = torch.zeros(*(shape[:-1] + [1]), dtype=updates_indices.dtype, device=updates.device)
        output_shape_ind = list(indices.shape[:-1]) + [1]
        ret_indices[slices] = updates_indices.view(*output_shape_ind)
        
        return ret, ret_indices
    
    return ret


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None):
        """
        Args:
            grid: pre-allocated grid tensor.
                  should be used when the volume of spatial shape
                  is very large.
        """
        self.features = features
        self.indices = indices
        if self.indices.dtype != torch.int32:
            self.indices.int()
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = grid

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(self, key):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    # def dense(self, channels_first=True):
    #     output_shape = (
    #         [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
    #     )
    #     res = scatter_nd(self.indices.long(), self.features, output_shape)
    #     if not channels_first:
    #         return res
    #     ndim = len(self.spatial_shape)
    #     trans_params = list(range(0, ndim + 1))
    #     trans_params.insert(1, ndim + 1)
    #     return res.permute(*trans_params).contiguous()
    
    def dense(self, channels_first=True):
        output_shape = (
            [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
        )
        res, res_ind = scatter_nd(self.indices.long(), self.features, output_shape, return_ind=True)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous(), res_ind.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return self.indices.shape[0] / np.prod(self.spatial_shape) / self.batch_size
