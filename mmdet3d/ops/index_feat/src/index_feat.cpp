// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations
void index_feat(int c, int n_intervals, int ref_num, 
    const float* feat, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, 
    int* attn_mask, float* out);

void index_feat_grad(int c, int n_intervals, int ref_num, const float* out_grad,
  const float* feat, const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths,
  const int* attn_mask, float* feat_grad);


/*
  Function: pillar pooling (forward, cuda)
  Args:
    feat             : input features, FloatTensor[n, h, w, c]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b*H*W, ref_num, c]
  Return:
*/
void index_feat_forward(
  const at::Tensor _feat,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  at::Tensor _attn_mask,
  at::Tensor _out
) {
  int c = _feat.size(4);
  int ref_num = _out.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feat));
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();

  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  int* attn_mask = _attn_mask.data_ptr<int>();
  float* out = _out.data_ptr<float>();
  
  index_feat(
    c, n_intervals, ref_num, feat, ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, attn_mask, out
  );
}


/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : grad of output bev feature, FloatTensor[b*H*W, ref_num, c]
    depth_grad       : grad of input depth, FloatTensor[n, d, h, w]
    feat_grad        : grad of input feature, FloatTensor[n, h, w, c]
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
*/
void index_feat_backward(
  const at::Tensor _out_grad,
  at::Tensor _feat_grad,
  const at::Tensor _feat,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  const at::Tensor _attn_mask
) {
  int c = _out_grad.size(2);
  int ref_num = _out_grad.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();

  float* feat_grad = _feat_grad.data_ptr<float>();
  const float* feat = _feat.data_ptr<float>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  const int* attn_mask = _attn_mask.data_ptr<int>();

  index_feat_grad(
    c, n_intervals, ref_num, out_grad, feat, 
    ranks_feat, ranks_bev, interval_starts, interval_lengths, 
    attn_mask, feat_grad
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_feat_forward", &index_feat_forward,
        "index_feat_forward");
  m.def("index_feat_backward", &index_feat_backward,
        "index_feat_backward");
}
