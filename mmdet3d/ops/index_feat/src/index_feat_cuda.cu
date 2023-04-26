// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>

/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b*H*W, ref_num, c]
*/
__global__ void index_feat_kernel(int c, int n_intervals, int ref_num,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  int *__restrict__ attn_mask,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  const float* cur_feat;
  float* cur_out;
  for(int i = 0; i < ref_num; i++){
    if (i >= interval_length) break;
    attn_mask[interval_start+i] = i;
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    cur_out = out + ranks_bev[interval_start] * ref_num * c + i * c + cur_c;
    *cur_out = *cur_feat;
  }

}


/*
  Function: pillar pooling backward
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    ref_num          : number of reference points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b*H*W, ref_num, c]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    feat_grad        : gradient of the feature fmap, FloatTensor, [b,n,h,w,c]
*/
__global__ void index_feat_grad_kernel(int c, int n_intervals, int ref_num,
                                  const float *__restrict__ out_grad,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  const int *__restrict__ attn_mask,
                                  float* __restrict__ feat_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];;

  // const int* cur_rank;
  const float* cur_out_grad;
  float* cur_feat_grad = feat_grad + ranks_feat[interval_start] * c + cur_c ;
  float grad_sum = 0;
  int idx_pts = 0;

  for(int i = 0; i < interval_length; i++){
    if (attn_mask[interval_start+i] == -1) continue;
    idx_pts = attn_mask[interval_start+i];
    // Same camera feature gradient from multiple BEV feature gradients
    // cur_rank = ranks_bev + interval_start + i;
    
    cur_out_grad = out_grad + ranks_bev[interval_start+i] * ref_num * c + idx_pts * c + cur_c;
    grad_sum += *cur_out_grad;
  }
  *cur_feat_grad = grad_sum;
}



void index_feat(int c, int n_intervals, int ref_num, 
  const float* feat, const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  int* attn_mask, float* out) {
  index_feat_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, ref_num, feat, ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, attn_mask, out
  );
}

void index_feat_grad(int c, int n_intervals, int ref_num, 
  const float* out_grad, const float* feat, const int* ranks_feat, const int* ranks_bev, 
  const int* interval_starts, const int* interval_lengths, 
  const int* attn_mask, float* feat_grad) {
  index_feat_grad_kernel<<<(int)ceil(((double)n_intervals *c / 256)), 256>>>(
     c, n_intervals, ref_num, out_grad, feat, 
     ranks_feat, ranks_bev, interval_starts, 
     interval_lengths, attn_mask, feat_grad
  );
}
