// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations

void depth_attn_qk2w(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* feat_bev, const float* feat_k, 
  const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  int* ranks_ref, float* out_weight);

void depth_attn_wk2q(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* feat_w, const float* feat_k, 
  const int* ranks_feat, const int* ranks_bev, const int* ranks_ref,
  const int* interval_starts, const int* interval_lengths, 
  float* out_bev);

void depth_attn_qw2k(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* feat_w, const float* feat_bev, 
  const int* ranks_feat, const int* ranks_bev,
  const int* interval_starts, const int* interval_lengths, 
  float* out_k);


/*
  Function: pillar pooling (forward, cuda)
  Args:
    feat_bev         : input depth, FloatTensor[b*H*W, c]
    feat_k           : input features, FloatTensor[b*n*h*w, c]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b*H*W, head_num, ref_num]
  Return:
*/

void depth_attn_weight_forward(
  const at::Tensor _feat_bev,
  const at::Tensor _feat_k,
  const at::Tensor _ranks_feat_f,
  const at::Tensor _ranks_bev_f,
  const at::Tensor _interval_lengths_f,
  const at::Tensor _interval_starts_f,
  at::Tensor _ranks_ref_f,
  at::Tensor _out_weight) {

  int c = _feat_k.size(1);
  int n_intervals = _interval_lengths_f.size(0);
  int ref_num = _out_weight.size(2);
  int head_num = _out_weight.size(1);
  int head_dim = c / head_num;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feat_bev));
  const float* feat_bev = _feat_bev.data_ptr<float>();
  const float* feat_k = _feat_k.data_ptr<float>();
  const int* ranks_feat = _ranks_feat_f.data_ptr<int>();
  const int* ranks_bev = _ranks_bev_f.data_ptr<int>();
  int* ranks_ref = _ranks_ref_f.data_ptr<int>();

  const int* interval_lengths = _interval_lengths_f.data_ptr<int>();
  const int* interval_starts = _interval_starts_f.data_ptr<int>();

  float* out_weight = _out_weight.data_ptr<float>();
  depth_attn_qk2w(
    c, n_intervals, ref_num, head_num, head_dim,
    feat_bev, feat_k,
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    ranks_ref, out_weight
  );
}



void depth_attn_weight_backward(
  const at::Tensor _out_w_grad,
  at::Tensor _bev_grad,
  at::Tensor _k_grad,
  const at::Tensor _feat_bev,
  const at::Tensor _feat_k,
  const at::Tensor _ranks_feat_b,
  const at::Tensor _ranks_bev_b,
  const at::Tensor _ranks_feat_f,
  const at::Tensor _ranks_bev_f,
  const at::Tensor _ranks_ref_b, 
  const at::Tensor _interval_lengths_b,
  const at::Tensor _interval_starts_b,
  const at::Tensor _interval_lengths_f,
  const at::Tensor _interval_starts_f) {

  int c = _feat_k.size(1);
  int n_intervals = _interval_lengths_b.size(0);
  int n_intervals_f = _interval_lengths_f.size(0);

  int ref_num = _out_w_grad.size(2);
  int head_num = _out_w_grad.size(1);
  int head_dim = c / head_num;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feat_bev));
  const float* out_w_grad = _out_w_grad.data_ptr<float>();

  float* bev_grad = _bev_grad.data_ptr<float>();
  float* k_grad = _k_grad.data_ptr<float>();

  const float* feat_bev = _feat_bev.data_ptr<float>();
  const float* feat_k = _feat_k.data_ptr<float>();

  const int* ranks_feat_b = _ranks_feat_b.data_ptr<int>();
  const int* ranks_bev_b = _ranks_bev_b.data_ptr<int>();
  const int* ranks_ref_b = _ranks_ref_b.data_ptr<int>();
  const int* interval_lengths_b = _interval_lengths_b.data_ptr<int>();
  const int* interval_starts_b = _interval_starts_b.data_ptr<int>();

  const int* ranks_feat_f = _ranks_feat_f.data_ptr<int>();
  const int* ranks_bev_f = _ranks_bev_f.data_ptr<int>();
  const int* interval_lengths_f = _interval_lengths_f.data_ptr<int>();
  const int* interval_starts_f = _interval_starts_f.data_ptr<int>();

  depth_attn_wk2q(
    c, n_intervals, ref_num, head_num, head_dim, 
    out_w_grad, feat_k, 
    ranks_feat_b, ranks_bev_b, ranks_ref_b,
    interval_starts_b, interval_lengths_b,
    bev_grad
  );

  depth_attn_qw2k(
    c, n_intervals_f, ref_num, head_num, head_dim,
    out_w_grad, feat_bev,
    ranks_feat_f, ranks_bev_f,
    interval_starts_f, interval_lengths_f, 
    k_grad
  );

}

/*
  Function: pillar pooling (forward, cuda)
  Args:
    weight         : input depth, FloatTensor[b*H*W, head_num, ref_num]
    feat_v             : input features, FloatTensor[b*n*h*w, c]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b*H*W, c]
  Return:
*/
void depth_attn_output_forward(
  const at::Tensor _weight,
  const at::Tensor _feat_v,
  const at::Tensor _ranks_feat_b,
  const at::Tensor _ranks_bev_b,
  const at::Tensor _ranks_ref_b,
  const at::Tensor _interval_lengths_b,
  const at::Tensor _interval_starts_b,
  at::Tensor _out) {

  int c = _feat_v.size(1);
  int n_intervals = _interval_lengths_b.size(0);
  int ref_num = _weight.size(2);
  int head_num = _weight.size(1);
  int head_dim = c / head_num;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_weight));
  const float* weight = _weight.data_ptr<float>();
  const float* feat_v = _feat_v.data_ptr<float>();
  const int* ranks_feat = _ranks_feat_b.data_ptr<int>();
  const int* ranks_bev = _ranks_bev_b.data_ptr<int>();
  const int* ranks_ref = _ranks_ref_b.data_ptr<int>();

  const int* interval_lengths = _interval_lengths_b.data_ptr<int>();
  const int* interval_starts = _interval_starts_b.data_ptr<int>();

  float* out = _out.data_ptr<float>();
  
  depth_attn_wk2q(
    c, n_intervals, ref_num, head_num, head_dim,
    weight, feat_v, 
    ranks_feat, ranks_bev, ranks_ref,
    interval_starts, interval_lengths, 
    out
  );
}



void depth_attn_output_backward(
  const at::Tensor _out_grad,
  at::Tensor _w_grad,
  at::Tensor _v_grad,
  const at::Tensor _weight,
  const at::Tensor _feat_v,
  const at::Tensor _ranks_feat_f,
  const at::Tensor _ranks_bev_f,
  at::Tensor _ranks_ref_f,
  const at::Tensor _interval_lengths_f,
  const at::Tensor _interval_starts_f) {

  int c = _feat_v.size(1);
  int n_intervals_f = _interval_lengths_f.size(0);
  int ref_num = _weight.size(2);
  int head_num = _weight.size(1);
  int head_dim = c / head_num;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_weight));
  const float* out_grad = _out_grad.data_ptr<float>();
  float* w_grad = _w_grad.data_ptr<float>();
  float* v_grad = _v_grad.data_ptr<float>();

  const float* weight = _weight.data_ptr<float>();
  const float* feat_v = _feat_v.data_ptr<float>();

  int* ranks_ref_f = _ranks_ref_f.data_ptr<int>();

  const int* ranks_feat_f = _ranks_feat_f.data_ptr<int>();
  const int* ranks_bev_f = _ranks_bev_f.data_ptr<int>();
  const int* interval_lengths_f = _interval_lengths_f.data_ptr<int>();
  const int* interval_starts_f = _interval_starts_f.data_ptr<int>();


  depth_attn_qk2w(
    c, n_intervals_f, ref_num, head_num, head_dim, 
    out_grad, feat_v, 
    ranks_feat_f, ranks_bev_f, 
    interval_starts_f, interval_lengths_f,
    ranks_ref_f, w_grad
  );

  depth_attn_qw2k(
    c, n_intervals_f, ref_num, head_num, head_dim,
    weight, out_grad,
    ranks_feat_f, ranks_bev_f, 
    interval_starts_f, interval_lengths_f, 
    v_grad
  );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_attn_weight_forward", &depth_attn_weight_forward,
        "depth_attn_weight_forward");
  m.def("depth_attn_weight_backward", &depth_attn_weight_backward,
        "depth_attn_weight_backward");
  m.def("depth_attn_output_forward", &depth_attn_output_forward,
        "depth_attn_output_forward");
  m.def("depth_attn_output_backward", &depth_attn_output_backward,
        "depth_attn_output_backward");
}
