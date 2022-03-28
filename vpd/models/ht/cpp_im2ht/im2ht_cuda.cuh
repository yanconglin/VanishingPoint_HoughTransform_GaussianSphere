#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#include "cuda_utils.h"

template <typename scalar_t>
__global__ void im2ht_cuda_forward_kernel(const int n,
                                  const scalar_t *data_im, 
                                  scalar_t *data_ht,
                                  const scalar_t *ht_index,
                                  const int batch, const int channel,
                                  const int height, const int width, 
                                  const int ht_height, const int ht_width,
                                  const int num_votes
                                )
{
  // launch channel*num_votes   cores
  // todo: coalesce
  CUDA_KERNEL_LOOP(index, n)
  {
  // input: [batch, channel, height, width]
  // ht_index: [num_votes, 3]
  // ht_index: [img_hw, ht_hw, ht_vote]
    int v = index % num_votes; // vote
    int c = index /num_votes % channel;  // channel

    int hw = ht_index[v * 3 + 0];
    int h = hw / width  % height;
    int w = hw % width;

    int ht_hw = ht_index[v * 3 + 1];
    int ht_h = ht_hw / ht_width  % ht_height;
    int ht_w = ht_hw % ht_width;

    scalar_t ht_vote = ht_index[v * 3 + 2];

    for (int b = 0; b < batch; ++b)
      {
        // input: [batch, channel, height, width]
        int offset_im = b * (channel * height * width) + c * (height * width) + h * width + w;
        // output: [batch, channel, ht_height, ht_width]
        int offset_ht =  b * (channel * ht_height * ht_width) + c * (ht_height * ht_width) + ht_h * ht_width + ht_w;

        scalar_t vote_value = data_im[offset_im];
        atomicAdd(data_ht+offset_ht, vote_value * ht_vote);
        // atomicAdd(data_ht+offset_ht, vote_value);
        // printf("b = %03d, c = %03d, h = %03d, w = %03d, rho_index = %03d, angle_idx = %03d\n", b, c, h, w, rho_index, angle_idx);
        // if (vote_value>0.0)
        // {
        //   printf("htindex = %5d, , vote_value = %.8f, ht_h = %3d, ht_w = %3d, ht_vote = %.8f\n", ht_hw, vote_value, ht_h, ht_w, ht_vote);
        //   // printf("imgindex = %5d, htindex = %5d, vote_value = %.8f, ht_vote = %.8f\n", hw, ht_hw, vote_value, ht_vote);
        // }
      }
  }
}




template <typename scalar_t>
__global__ void im2ht_cuda_backward_kernel(const int n,
                                    scalar_t* grad_im,
                                    const scalar_t* grad_ht, 
                                    const scalar_t* ht_index,
                                    const int batch, const int channel,
                                    const int height, const int width, 
                                    const int ht_height, const int ht_width,
                                    const int num_votes
                                  )
{
  // launch channel * height *  ht_w  cores
  // todo: coalesce
  CUDA_KERNEL_LOOP(index, n)
  {
    // input: [batch, channel, height, width]
    // ht_index: [num_votes, 3]
    // ht_index: [img_hw, ht_hw, ht_vote]
    int v = index % num_votes; // vote
    int c = index /num_votes % channel;  // channel

    int hw = ht_index[v * 3 + 0];
    int h = hw / width  % height;
    int w = hw % width;

    int ht_hw = ht_index[v * 3 + 1];
    int ht_h = ht_hw / ht_width  % ht_height;
    int ht_w = ht_hw % ht_width;

    scalar_t ht_vote = ht_index[v * 3 + 2];

    for (int b = 0; b < batch; ++b)
    {
      // input: [batch, channel, height, width]
      int offset_im = b * (channel * height * width) + c * (height * width) + h * width + w;
      // output: [batch, channel, ht_height, ht_width]
      int offset_ht =  b * (channel * ht_height * ht_width) + c * (ht_height * ht_width) + ht_h * ht_width + ht_w;

      scalar_t grad_value = grad_ht[offset_ht];
      atomicAdd(grad_im + offset_im, grad_value*ht_vote);
      // atomicAdd(grad_im + offset_im, grad_value);
    }
  }
}


template <typename scalar_t>
void im2ht_cuda_forward(cudaStream_t stream,
                  const scalar_t* data_im, 
                  scalar_t* data_ht,
                  const scalar_t* ht_index,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int ht_height, const int ht_width,
                  const int num_votes
                ) 
{
  const int num_kernels = channel* num_votes;
  im2ht_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          data_im, 
                                                          data_ht,
                                                          ht_index, 
                                                          batch, channel, 
                                                          height, width, 
                                                          ht_height, ht_width,
                                                          num_votes
                                                        );
                                                      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in im2ht_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void im2ht_cuda_backward(cudaStream_t stream,
                  scalar_t* grad_im,
                  const scalar_t* grad_ht, 
                  const scalar_t* ht_index,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int ht_height, const int ht_width,
                  const int num_votes
                  )
{

  const int num_kernels = channel* num_votes;
  // ***********************************************************************//
  im2ht_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_im, 
                                                          grad_ht, 
                                                          ht_index,
                                                          batch, channel, 
                                                          height, width,
                                                          ht_height, ht_width,
                                                          num_votes
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ht2im_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

