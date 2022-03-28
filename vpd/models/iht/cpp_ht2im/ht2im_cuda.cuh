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
__global__ void ht2im_cuda_forward_kernel(const int n,
                                    const scalar_t* data_ht,
                                    scalar_t* data_im, 
                                    const scalar_t* ht_index,
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
        scalar_t ht_value = data_ht[offset_ht];
        // atomicAdd(data_im + offset_im, ht_value*ht_vote);
        atomicAdd(data_im + offset_im, ht_value);
      }
  }
}





template <typename scalar_t>
__global__ void ht2im_cuda_backward_kernel(const int n,
                                  scalar_t *grad_ht, 
                                  const scalar_t *grad_im,
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
        scalar_t grad_value = grad_im[offset_im];
        // atomicAdd(grad_ht+offset_ht, grad_value*ht_vote);
        atomicAdd(grad_ht+offset_ht, grad_value);
        // printf("b = %03d, c = %03d, h = %03d, w = %03d, offset_img = %05d\n", b, c, h, w, offset_img);
        // printf("Imgindex = %5d, vote_value = %.8f, offset = %5d\n", offset_img, vote_value, offset);
      }
  }
}



template <typename scalar_t>
void ht2im_cuda_forward(cudaStream_t stream,
                  const scalar_t* data_ht, 
                  scalar_t* data_im,
                  const scalar_t* ht_index,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int ht_height, const int ht_width,
                  const int num_votes
                ) 
{
  const int num_kernels = channel* num_votes;
  ht2im_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          data_ht, 
                                                          data_im,
                                                          ht_index, 
                                                          batch, channel, 
                                                          height, width, 
                                                          ht_height, ht_width,
                                                          num_votes
                                                        );
                                                      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ht2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void ht2im_cuda_backward(cudaStream_t stream,
                  scalar_t* grad_ht,
                  const scalar_t* grad_im, 
                  const scalar_t* ht_index,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int ht_height, const int ht_width,
                  const int num_votes
                  )
{

  const int num_kernels = channel* num_votes;
  // ***********************************************************************//
  ht2im_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_ht, 
                                                          grad_im, 
                                                          ht_index,
                                                          batch, channel, 
                                                          height, width, 
                                                          ht_height, ht_width,
                                                          num_votes
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in im2ht_cuda: %s\n", cudaGetErrorString(err));
  }

}

