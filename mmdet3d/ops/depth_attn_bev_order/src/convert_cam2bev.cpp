// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations

void convert_bev2cam(int n_intervals, int ref_num_bev, int ref_num_cam, 
                      int head_num, int head_dim,
                      const float* input_bev, const int* ranks_ref,
                      const int* ranks_feat, const int* ranks_bev, 
                      const int* interval_starts, const int* interval_lengths,
                      float* output_cam);

void convert_cam2bev(int n_intervals, int ref_num_bev, int ref_num_cam, 
                      int head_num, int head_dim,
                      const float* input_cam, const int* ranks_ref,
                      const int* ranks_feat, const int* ranks_bev, 
                      const int* interval_starts, const int* interval_lengths, 
                      float* output_bev);



void convert_bev2cam_(
  const at::Tensor _in_bev,
  const at::Tensor _ranks_ref,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  at::Tensor _out_cam) {


  int n_intervals = _interval_lengths.size(0);
  int head_num = _in_bev.size(1);
  int ref_num_bev = _in_bev.size(2);
  int ref_num_cam = _out_cam.size(2)

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_bev));
  const float* in_bev = _in_bev.data_ptr<float>();

  const int* ranks_ref = _ranks_ref.data_ptr<int>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();

  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  float* out_cam = _out_cam.data_ptr<float>();
  depth_attn_weight(
    c, n_intervals, ref_num, head_num, head_dim,
    feat_bev, feat_k,
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    key_padding_mask, q_padding_mask, out_weight
  );

}

void convert_cam2bev_() {

}

void depth_attn_weight_forward(
  const at::Tensor _feat_bev,
  const at::Tensor _feat_k,
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  at::Tensor _key_padding_mask,
  at::Tensor _q_padding_mask,
  at::Tensor _out_weight) {
  int c = _feat_k.size(1);
  int n_intervals = _interval_lengths.size(0);
  int ref_num = _out_weight.size(2);
  int head_num = _out_weight.size(1);
  int head_dim = c / head_num;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feat_bev));
  const float* feat_bev = _feat_bev.data_ptr<float>();
  const float* feat_k = _feat_k.data_ptr<float>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();

  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  int* key_padding_mask = _key_padding_mask.data_ptr<int>();
  bool* q_padding_mask = _q_padding_mask.data_ptr<bool>();
  float* out_weight = _out_weight.data_ptr<float>();
  depth_attn_weight(
    c, n_intervals, ref_num, head_num, head_dim,
    feat_bev, feat_k,
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    key_padding_mask, q_padding_mask, out_weight
  );
}



void depth_attn_weight_backward(
  const at::Tensor _out_grad,
  at::Tensor _bev_grad,
  at::Tensor _k_grad,
  const at::Tensor _feat_bev,
  const at::Tensor _feat_k,
  const at::Tensor _ranks_feat_b,
  const at::Tensor _ranks_bev_b,
  const at::Tensor _ranks_feat_f,
  const at::Tensor _ranks_bev_f,
  const at::Tensor _interval_lengths_b,
  const at::Tensor _interval_starts_b,
  const at::Tensor _interval_lengths_f,
  const at::Tensor _interval_starts_f,
  const at::Tensor _key_padding_mask_f) {
  int c = _feat_k.size(1);
  int n_intervals = _interval_lengths_b.size(0);
  int n_intervals_f = _interval_lengths_f.size(0);

  int ref_num = _out_grad.size(2);
  int head_num = _out_grad.size(1);
  int head_dim = c / head_num;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feat_bev));
  const float* out_grad = _out_grad.data_ptr<float>();
  float* bev_grad = _bev_grad.data_ptr<float>();
  float* k_grad = _k_grad.data_ptr<float>();

  const float* feat_bev = _feat_bev.data_ptr<float>();
  const float* feat_k = _feat_k.data_ptr<float>();


  const int* ranks_feat_b = _ranks_feat_b.data_ptr<int>();
  const int* ranks_bev_b = _ranks_bev_b.data_ptr<int>();
  const int* interval_lengths_b = _interval_lengths_b.data_ptr<int>();
  const int* interval_starts_b = _interval_starts_b.data_ptr<int>();

  const int* ranks_feat_f = _ranks_feat_f.data_ptr<int>();
  const int* ranks_bev_f = _ranks_bev_f.data_ptr<int>();
  const int* interval_lengths_f = _interval_lengths_f.data_ptr<int>();
  const int* interval_starts_f = _interval_starts_f.data_ptr<int>();

  const int* key_padding_mask_f = _key_padding_mask_f.data_ptr<int>();

  depth_attn_weight_grad_bev(
    c, n_intervals, ref_num, head_num, head_dim, 
    out_grad, feat_k, 
    ranks_feat_b, ranks_bev_b, interval_starts_b, interval_lengths_b,
    bev_grad
  );

  depth_attn_weight_grad_k(
    c, n_intervals_f, ref_num, head_num, head_dim,
    out_grad, feat_bev,
    ranks_feat_f, ranks_bev_f, interval_starts_f, interval_lengths_f, 
    key_padding_mask_f, k_grad
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
  const at::Tensor _ranks_feat,
  const at::Tensor _ranks_bev,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  at::Tensor _out) {
  int c = _feat_v.size(1);
  int n_intervals = _interval_lengths.size(0);
  int ref_num = _weight.size(2);
  int head_num = _weight.size(1);
  int head_dim = c / head_num;
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_weight));
  const float* weight = _weight.data_ptr<float>();
  const float* feat_v = _feat_v.data_ptr<float>();
  const int* ranks_feat = _ranks_feat.data_ptr<int>();
  const int* ranks_bev = _ranks_bev.data_ptr<int>();

  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  float* out = _out.data_ptr<float>();
  depth_attn_output(
    c, n_intervals, ref_num, head_num, head_dim,
    feat_v, weight,
    ranks_feat, ranks_bev, 
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
  const at::Tensor _ranks_feat_b,
  const at::Tensor _ranks_bev_b,
  const at::Tensor _ranks_feat_f,
  const at::Tensor _ranks_bev_f,
  const at::Tensor _interval_lengths_b,
  const at::Tensor _interval_starts_b,
  const at::Tensor _interval_lengths_f,
  const at::Tensor _interval_starts_f,
  const at::Tensor _key_padding_mask_f) {
  int c = _feat_v.size(1);
  int n_intervals = _interval_lengths_b.size(0);
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

  const int* ranks_feat_b = _ranks_feat_b.data_ptr<int>();
  const int* ranks_bev_b = _ranks_bev_b.data_ptr<int>();
  const int* interval_lengths_b = _interval_lengths_b.data_ptr<int>();
  const int* interval_starts_b = _interval_starts_b.data_ptr<int>();

  const int* ranks_feat_f = _ranks_feat_f.data_ptr<int>();
  const int* ranks_bev_f = _ranks_bev_f.data_ptr<int>();
  const int* interval_lengths_f = _interval_lengths_f.data_ptr<int>();
  const int* interval_starts_f = _interval_starts_f.data_ptr<int>();

  const int* key_padding_mask_f = _key_padding_mask_f.data_ptr<int>();

  depth_attn_output_grad_w(
    c, n_intervals, ref_num, head_num, head_dim, 
    out_grad, feat_v, 
    ranks_feat_b, ranks_bev_b, interval_starts_b, interval_lengths_b,
    w_grad
  );

  depth_attn_output_grad_v(
    c, n_intervals_f, ref_num, head_num, head_dim,
    out_grad, weight,
    ranks_feat_f, ranks_bev_f, interval_starts_f, interval_lengths_f, 
    key_padding_mask_f, v_grad
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
