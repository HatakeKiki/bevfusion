// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>


/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    feat_bev         : input depth, FloatTensor[b*h*w, c]
    feat_k           : input feat, FloatTensor[b*N*h*w,c]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b*H*W, ref_num, num_head]
*/
__global__ void depth_attn_weight_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                                  const float *__restrict__ feat_bev,
                                  const float *__restrict__ feat_k,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  int *__restrict__ key_padding_mask,
                                  bool *__restrict__ q_padding_mask,
                                  float *__restrict__ out_weight) {

  int idx_interval = blockIdx.x;
  int idx_head = blockIdx.y;
  int idx_pts = threadIdx.x;

  if (idx_interval >= n_intervals) return;
  if (idx_pts >= interval_lengths[idx_interval]) return;
  if (idx_head >= head_num) return;

  int interval_start = interval_starts[idx_interval];
  int idx_ranks = interval_start + idx_pts;
  int cur_c = idx_head * head_dim;

  // key_padding_mask for reference point record
  if (idx_head == 0) {
    key_padding_mask[idx_ranks] = idx_pts;
    q_padding_mask[ranks_bev[idx_ranks]] = false;
  }
  
  const float* cur_feat_k = feat_k + ranks_feat[idx_ranks] * c + cur_c;
  const float* cur_feat_bev = feat_bev + ranks_bev[idx_ranks] * c + cur_c;

  float* cur_out = out_weight + ranks_bev[idx_ranks] * head_num * ref_num + idx_head * ref_num + idx_pts;
  float psum = 0;
  for(int i = 0; i < head_dim; i++) {
    psum += *cur_feat_bev * *cur_feat_k;
    cur_feat_k++;
    cur_feat_bev++;
  }
  *cur_out = psum;
}

__global__ void depth_attn_weight_grad_bev_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                                              const float *__restrict__ out_grad, 
                                              const float *__restrict__ feat_k,
                                              const int *__restrict__ ranks_feat, 
                                              const int *__restrict__ ranks_bev,
                                              const int *__restrict__ interval_starts, 
                                              const int *__restrict__ interval_lengths,
                                              float *__restrict__ bev_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  int idx_head = cur_c / head_dim;

  const float* cur_feat_k;
  const float* cur_out_grad = out_grad 
                              + ranks_bev[interval_start] * head_num * ref_num 
                              + idx_head * ref_num;

  float* cur_out = bev_grad + ranks_bev[interval_start] * c + cur_c;
  float psum = 0;

  for (int i = 0; i < ref_num; i++) {
    if (i >= interval_length) break;
    cur_feat_k = feat_k + ranks_feat[interval_start+i] * c + cur_c;
    psum += *cur_feat_k * *cur_out_grad;
    cur_out_grad++;
  }
  *cur_out = psum;
}

__global__ void depth_attn_weight_grad_k_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim, 
                                                const float *__restrict__ out_grad, 
                                                const float *__restrict__ feat_bev, 
                                                const int *__restrict__ ranks_feat, 
                                                const int *__restrict__ ranks_bev,
                                                const int *__restrict__ interval_starts, 
                                                const int *__restrict__ interval_lengths,
                                                const int *__restrict__ key_padding_mask, 
                                                float *__restrict__ k_grad) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  int idx_head = cur_c / head_dim;

  const float* cur_feat_bev;
  const float* cur_out_grad;

  float* cur_out = k_grad + ranks_feat[interval_start] * c + cur_c;
  int idx_ranks;
  float psum = 0;
  for (int i = 0; i < interval_length; i++) {
    idx_ranks = interval_start + i;
    if (key_padding_mask[idx_ranks] == -1) continue;

    cur_feat_bev = feat_bev + ranks_bev[idx_ranks] * c + cur_c;
    cur_out_grad = out_grad 
                   + ranks_bev[idx_ranks] * head_num * ref_num + idx_head * ref_num 
                   + key_padding_mask[idx_ranks];

    psum += *cur_feat_bev * *cur_out_grad;
  }

  *cur_out = psum;
}

__global__ void depth_attn_output_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                                         const float *__restrict__ feat_v, 
                                         const float *__restrict__ weight, 
                                         const int *__restrict__ ranks_feat, 
                                         const int *__restrict__ ranks_bev, 
                                         const int *__restrict__ interval_starts, 
                                         const int *__restrict__ interval_lengths, 
                                         float* __restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  int idx_head = cur_c / head_dim;

  const float* cur_feat_v;
  const float * cur_weight = weight 
                            + ranks_bev[interval_start] * head_num * ref_num 
                            + idx_head * ref_num;

  float* cur_output = output + ranks_bev[interval_start] * c + cur_c;
  float psum = 0;

  for (int i = 0; i < ref_num; i++) {
    if (i >= interval_length) break;
    cur_feat_v = feat_v + ranks_feat[interval_start + i] * c + cur_c;
    psum += *cur_weight * *cur_feat_v;
    cur_weight++;

  }

  *cur_output = psum;
 }

 __global__ void depth_attn_output_grad_w_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                                                const float *__restrict__ out_grad, 
                                                const float *__restrict__ feat_v, 
                                                const int *__restrict__ ranks_feat, 
                                                const int *__restrict__ ranks_bev, 
                                                const int *__restrict__ interval_starts, 
                                                const int *__restrict__ interval_lengths, 
                                                float *__restrict__ w_grad) {

  int idx_interval = blockIdx.x;
  int idx_head = blockIdx.y;
  int idx_pts = threadIdx.x;

  if (idx_interval >= n_intervals) return;
  if (idx_pts >= interval_lengths[idx_interval]) return;
  if (idx_head >= head_num) return;

  int interval_start = interval_starts[idx_interval];
  int idx_ranks = interval_start + idx_pts;
  int cur_c = idx_head * head_dim;

  const float* cur_feat_v = feat_v + ranks_feat[idx_ranks] * c + cur_c;
  const float* cur_out_grad = out_grad + ranks_bev[idx_ranks] * c + cur_c;

  float* cur_out = w_grad + ranks_bev[idx_ranks] * head_num * ref_num + idx_head * ref_num + idx_pts;
  float psum = 0;
  for (int i = 0; i < head_dim; i++) {
    psum += *cur_feat_v * * cur_out_grad;
    cur_feat_v++;
    cur_out_grad++;
  }
  *cur_out = psum;
 }

 __global__ void depth_attn_output_grad_v_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                                                const float *__restrict__ out_grad, 
                                                const float *__restrict__ weight, 
                                                const int *__restrict__ ranks_feat, 
                                                const int *__restrict__ ranks_bev, 
                                                const int *__restrict__ interval_starts, 
                                                const int *__restrict__ interval_lengths, 
                                                const int *__restrict__ key_padding_mask, 
                                                float *__restrict__ v_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  int idx_head = cur_c / head_dim;

  const float* cur_weight;
  const float* cur_out_grad;
  // const int* cur_key_padding_mask;

  float* cur_grad_v = v_grad + ranks_feat[interval_start] * c + cur_c;
  float psum = 0;
  int idx_ranks = 0;
  for (int i = 0; i < interval_length; i++) {
    idx_ranks = interval_start + i;
    if (key_padding_mask[idx_ranks] == -1) continue;
    cur_weight = weight 
                 + ranks_bev[idx_ranks] * head_num * ref_num 
                 + idx_head * ref_num + key_padding_mask[idx_ranks];
    cur_out_grad = out_grad + ranks_bev[idx_ranks]*c + cur_c;
    psum += *cur_weight * *cur_out_grad;
  }
  *cur_grad_v = psum;
 }



void depth_attn_weight(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                      const float* feat_bev, const float* feat_k, 
                      const int* ranks_feat, const int* ranks_bev, 
                      const int* interval_starts, const int* interval_lengths, 
                      int* key_padding_mask, bool* q_padding_mask, float* out_weight) {

  dim3 grid(n_intervals, head_num, 1);
  
  depth_attn_weight_kernel<<<grid, ref_num>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    feat_bev, feat_k, 
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    key_padding_mask, q_padding_mask, out_weight
  );
}

void depth_attn_weight_grad_bev(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* out_grad, const float* feat_k, 
  const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  float* bev_grad
) {
  depth_attn_weight_grad_bev_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    out_grad, feat_k, 
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    bev_grad
  );
}

void depth_attn_weight_grad_k(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* out_grad, const float* feat_bev, 
  const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  const int* key_padding_mask, float* k_grad
) {
  depth_attn_weight_grad_k_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    out_grad, feat_bev, 
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    key_padding_mask, k_grad
  );
}

void depth_attn_output(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* feat_v, const float* weight, 
  const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  float* output
) {
  depth_attn_output_kernel<<<(int)ceil(((double)n_intervals  * c / 256)), 256>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    feat_v, weight, 
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    output
  );
}

void depth_attn_output_grad_w(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* out_grad, const float* feat_v, 
  const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  float* w_grad
) {
  dim3 grid(n_intervals, head_num);
  depth_attn_output_grad_w_kernel<<<grid, ref_num>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    out_grad, feat_v, 
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    w_grad
  );
}

void depth_attn_output_grad_v(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* out_grad, const float* weight, 
  const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  const int* key_padding_mask, float* v_grad) {
  depth_attn_output_grad_v_kernel<<<(int)ceil(((double)n_intervals  * c / 256)), 256>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    out_grad, weight, 
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    key_padding_mask, v_grad
  );
}
