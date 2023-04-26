// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>


/*
  Function: depth_attn_weight_forward
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    feat_bev         : input depth, FloatTensor[b*h*w, c]
    feat_k           : input feat, FloatTensor[b*N*h*w,c]

    ## ranks in feat order
    ranks_ref
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]

    out              : output features, FloatTensor[b*N*h*w, num_head, ref_num]

    used in weight and output_grad_w (qk2w)
*/
__global__ void depth_attn_qk2w_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                                      const float *__restrict__ feat_bev,
                                      const float *__restrict__ feat_k,
                                      const int *__restrict__ ranks_feat,
                                      const int *__restrict__ ranks_bev,
                                      const int *__restrict__ interval_starts,
                                      const int *__restrict__ interval_lengths,
                                      int *__restrict__ ranks_ref,
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

  if ((idx_head == 0) && (ranks_ref[idx_ranks] == -1)) ranks_ref[idx_ranks] = idx_pts;

  const float* cur_feat_k = feat_k + ranks_feat[idx_ranks] * c + cur_c;
  const float* cur_feat_bev = feat_bev + ranks_bev[idx_ranks] * c + cur_c;

  float* cur_out_weight = out_weight 
                          + ranks_feat[idx_ranks] * head_num * ref_num 
                          + idx_head * ref_num + idx_pts;
  float psum = 0;
  // if (*cur_out_weight > 0) psum = *cur_out_weight;
  for(int i = 0; i < head_dim; i++) {
    psum += *cur_feat_bev * *cur_feat_k;
    cur_feat_k++;
    cur_feat_bev++;
  }
  *cur_out_weight = psum;
}

/*
  Function: depth_attn_weight_grad_bev
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : input depth, FloatTensor[b*N*h*w, head_num, ref_num]
    feat_k           : input feat, FloatTensor[b*N*h*w,c]

    ## ranks in bev order
    ranks_ref
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]

    bev_grad         : output features, FloatTensor[b*H*W, c]

    used in weight_grad_q and output (wv2q)
*/
__global__ void depth_attn_wk2q_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim,
                                      const float *__restrict__ feat_w, 
                                      const float *__restrict__ feat_k,
                                      const int *__restrict__ ranks_feat, 
                                      const int *__restrict__ ranks_bev,
                                      const int *__restrict__ ranks_ref,
                                      const int *__restrict__ interval_starts, 
                                      const int *__restrict__ interval_lengths,
                                      float *__restrict__ out_bev) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  int idx_head = cur_c / head_dim;

  const float* cur_feat_k;
  const float* cur_feat_w;

  float* cur_out_bev = out_bev + ranks_bev[interval_start] * c + cur_c;
  int idx_ranks;
  float psum = 0;

  for (int i = 0; i < interval_length; i++) {
    idx_ranks = interval_start + i;
    if (ranks_ref[idx_ranks] == -1) continue;

    cur_feat_k = feat_k + ranks_feat[interval_start+i] * c + cur_c;
    cur_feat_w = feat_w 
                + ranks_feat[idx_ranks] * head_num * ref_num 
                + idx_head * ref_num 
                + ranks_ref[idx_ranks];
    
    psum += *cur_feat_k * *cur_feat_w;
  }
  *cur_out_bev = psum;
}

/*
used in weight_grad_k and output_grad_v
*/
__global__ void depth_attn_qw2k_kernel(int c, int n_intervals, int ref_num, int head_num, int head_dim, 
                                      const float *__restrict__ feat_w, 
                                      const float *__restrict__ feat_bev, 
                                      const int *__restrict__ ranks_feat, 
                                      const int *__restrict__ ranks_bev,
                                      const int *__restrict__ interval_starts, 
                                      const int *__restrict__ interval_lengths, 
                                      float *__restrict__ out_k) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  int idx_head = cur_c / head_dim;

  const float* cur_feat_bev;
  const float* cur_feat_w = feat_w 
                              + ranks_feat[interval_start] * head_num * ref_num 
                              + idx_head * ref_num;

  float* cur_out_k = out_k + ranks_feat[interval_start] * c + cur_c;
  float psum = 0;
  for (int i = 0; i < interval_length; i++) {
    if (i >= ref_num) break;
    cur_feat_bev = feat_bev + ranks_bev[interval_start+i] * c + cur_c;
    psum += *cur_feat_bev * *cur_feat_w;
    cur_feat_w++;
  }
  *cur_out_k = psum;
}


void depth_attn_qk2w(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* feat_bev, const float* feat_k, 
  const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  int* ranks_ref, float* out_weight) {

  dim3 grid(n_intervals, head_num, 1);
  
  depth_attn_qk2w_kernel<<<grid, ref_num>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    feat_bev, feat_k, 
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    ranks_ref, out_weight
  );
}

void depth_attn_wk2q(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* feat_w, const float* feat_k, 
  const int* ranks_feat, const int* ranks_bev, const int* ranks_ref,
  const int* interval_starts, const int* interval_lengths, 
  float* out_bev) {
  depth_attn_wk2q_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    feat_w, feat_k, 
    ranks_feat, ranks_bev, ranks_ref,
    interval_starts, interval_lengths, 
    out_bev
  );
}

void depth_attn_qw2k(
  int c, int n_intervals, int ref_num, int head_num, int head_dim,
  const float* feat_w, const float* feat_bev, 
  const int* ranks_feat, const int* ranks_bev,
  const int* interval_starts, const int* interval_lengths, 
  float* out_k) {
  depth_attn_qw2k_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, ref_num, head_num, head_dim, 
    feat_w, feat_bev, 
    ranks_feat, ranks_bev,
    interval_starts, interval_lengths, 
    out_k
  );
}