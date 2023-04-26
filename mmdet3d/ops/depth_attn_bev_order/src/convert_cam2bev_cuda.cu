// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111

#include <stdio.h>
#include <stdlib.h>


/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    input_bev        : input depth, FloatTensor[b*H*W, num_head, ref_num]
    ranks_ref        : max -> ref_num_bev 
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    output_cam       : output features, FloatTensor[b*N*h*w, num_head, ref_num]
*/
__global__ void convert_bev2cam_kernel(int n_intervals, int ref_num_bev, int ref_num_cam, 
                                      int head_num, int head_dim,
                                      const float *__restrict__ input_bev,
                                      const int *__restrict__ ranks_ref,
                                      const int *__restrict__ ranks_feat,
                                      const int *__restrict__ ranks_bev,
                                      const int *__restrict__ interval_starts,
                                      const int *__restrict__ interval_lengths,
                                      float *__restrict__ output_cam) {

  int idx_interval = blockIdx.x;
  int idx_head = blockIdx.y;
  int idx_pts = threadIdx.x;

  if (idx_interval >= n_intervals) return;
  if (idx_pts >= interval_lengths[idx_interval]) return;
  if (idx_head >= head_num) return;

  int interval_start = interval_starts[idx_interval];
  int idx_ranks = interval_start + idx_pts;

  if (ranks_ref[idx_ranks] == -1) return;


  const float* cur_input_bev = input_bev + 
                                ranks_bev[idx_ranks] * head_num * ref_num_bev + 
                                idx_head * ref_num_bev + ranks_ref[idx_ranks];

  float* cur_output_cam = output_cam + 
                          ranks_feat[idx_ranks] * head_num * ref_num_cam +
                          idx_head * ref_num_cam + idx_pts;

  *cur_output_cam = *cur_input_bev;
}

__global__ void convert_cam2bev_kernel(int n_intervals, int ref_num_bev, int ref_num_cam, 
                                      int head_num, int head_dim,
                                      const float *__restrict__ input_cam,
                                      const int *__restrict__ ranks_ref,
                                      const int *__restrict__ ranks_feat,
                                      const int *__restrict__ ranks_bev,
                                      const int *__restrict__ interval_starts,
                                      const int *__restrict__ interval_lengths,
                                      float *__restrict__ output_bev) {

  int idx_interval = blockIdx.x;
  int idx_head = blockIdx.y;
  int idx_pts = threadIdx.x;

  if (idx_interval >= n_intervals) return;
  if (idx_pts >= interval_lengths[idx_interval]) return;
  if (idx_head >= head_num) return;

  int interval_start = interval_starts[idx_interval];
  int idx_ranks = interval_start + idx_pts;

  if (ranks_ref[idx_ranks] == -1) return;


  const float* cur_input_cam = input_bev + 
                                ranks_bev[idx_ranks] * head_num * ref_num_bev + 
                                idx_head * ref_num_bev + idx_pts;

  float* cur_output_bev = output_cam + 
                          ranks_feat[idx_ranks] * head_num * ref_num_cam +
                          idx_head * ref_num_cam + ranks_ref[idx_ranks];

  *cur_output_bev = *cur_input_cam;
}


void convert_bev2cam(int n_intervals, int ref_num_bev, int ref_num_cam, 
                      int head_num, int head_dim,
                      const float* input_bev, const int* ranks_ref,
                      const int* ranks_feat, const int* ranks_bev, 
                      const int* interval_starts, const int* interval_lengths,
                      float* output_cam) {

  dim3 grid(n_intervals, head_num, 1);
  
  convert_bev2cam_kernel<<<grid, ref_num_cam>>>(
    n_intervals, ref_num_bev, ref_num_cam, head_num, head_dim, 
    input_bev, ranks_ref,
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    output_cam
  );
}

/*

n_intervals, ref_num, ranks, intervals, attn_mask in bev-perspective

*/
void convert_cam2bev(int n_intervals, int ref_num_bev, int ref_num_cam, 
                    int head_num, int head_dim,
                    const float* input_cam, const int* ranks_ref,
                    const int* ranks_feat, const int* ranks_bev, 
                    const int* interval_starts, const int* interval_lengths, 
                    float* output_bev) {
  dim3 grid(n_intervals, head_num, 1);

  convert_cam2bev_kernel<<<grid, ref_num_bev>>>(
    n_intervals, ref_num_bev, ref_num_cam, head_num, head_dim, 
    input_cam, ranks_ref,
    ranks_feat, ranks_bev, 
    interval_starts, interval_lengths, 
    output_bev
  );
}
