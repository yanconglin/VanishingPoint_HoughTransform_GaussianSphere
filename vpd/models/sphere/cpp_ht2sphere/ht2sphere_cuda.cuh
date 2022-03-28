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
__global__ void ht2sphere_cuda_forward_kernel(const int n,
                                  const scalar_t *data_ht, 
                                  scalar_t *data_sphere,
                                  const scalar_t *sphere_index,
                                  const int batch, const int channel,
                                  const int height, const int width,
                                  const int sphere_size, const int num_votes
                                )
{
  // launch channel * num_votes  cores
  // todo: coalesce
  CUDA_KERNEL_LOOP(index, n)
  {
  // sphere_index: [num_votes, 3]
    int v = index % num_votes; // vote
    int c = index /num_votes % channel;  // channel
    // sphere_index: [hw, sphere_idx, sphere_vote]
    int hw = sphere_index[v * 3 + 0];
    int h = hw / width  % height;
    int w = hw % width;
    int sphere_idx = sphere_index[v * 3 + 1];
    scalar_t sphere_vote = sphere_index[v * 3 + 2];

    // output: [batch, channel, sphere_size]
    for (int b = 0; b < batch; ++b)
    { 
      if (sphere_idx>=0 && sphere_vote>0.0)
      {
        int offset_ht = b * (channel * height * width) + c * (height * width) + h * width + w;
        int offset_sphere =  b * (channel * sphere_size) + c * sphere_size + sphere_idx; // offset in the output (sphere)
        scalar_t vote_value = data_ht[offset_ht];
        atomicAdd(data_sphere+offset_sphere, vote_value*sphere_vote);
        // atomicAdd(data_sphere+offset_sphere, vote_value);
        // printf("b = %03d, c = %03d, h = %03d, w = %03d, offset_img = %05d\n", b, c, h, w, offset_img);
        // printf("Imgindex = %5d, vote_value = %.8f, offset = %5d\n", offset_img, vote_value, offset);
      }
    }
  }
}




template <typename scalar_t>
__global__ void ht2sphere_cuda_backward_kernel(const int n,
                                    scalar_t* grad_ht,
                                    const scalar_t* grad_sphere, 
                                    const scalar_t* sphere_index,
                                    const int batch, const int channel,
                                    const int height, const int width, 
                                    const int sphere_size, const int num_votes
                                  )
{
  // launch channel * num_votes  cores
  // todo: coalesce
  CUDA_KERNEL_LOOP(index, n)
  {
  // sphere_index: [num_votes, 3]
    int v = index % num_votes; // vote
    int c = index / num_votes % channel;  // channel
 
    // sphere_index: [hw, sphere_idx, sphere_vote]
    int hw = sphere_index[v * 3 + 0];
    int h = hw / width  % height;
    int w = hw % width;
    int sphere_idx = sphere_index[v * 3 + 1];
    scalar_t sphere_vote = sphere_index[v * 3 + 2];

    // grad_sphere: [batch, channel, sphere_size]
    // grad_ht: [batch, channel, height, width]
    for (int b = 0; b < batch; ++b)
    {
      if (sphere_idx>=0 && sphere_vote>0.0)
      {

        int offset_ht = b * (channel * height * width) + c * (height * width) + h * width + w;
        int offset_sphere =  b * (channel * sphere_size) + c * sphere_size + sphere_idx; // offset in the output (sphere)
        scalar_t grad_value = grad_sphere[offset_sphere];
        atomicAdd(grad_ht + offset_ht, grad_value*sphere_vote);
        // atomicAdd(grad_ht + offset_ht, grad_value);
      }
    }
  }
}



template <typename scalar_t>
void ht2sphere_cuda_forward(cudaStream_t stream,
                  const scalar_t* data_ht, 
                  scalar_t* data_sphere,
                  const scalar_t* sphere_index,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int sphere_size, const int num_votes
                ) 
{
  const int num_kernels = channel* num_votes;
  ht2sphere_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          data_ht, 
                                                          data_sphere,
                                                          sphere_index, 
                                                          batch, channel, 
                                                          height, width, 
                                                          sphere_size, num_votes
                                                        );
                                                      
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ht2sphere_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void ht2sphere_cuda_backward(cudaStream_t stream,
                  scalar_t* grad_ht,
                  const scalar_t* grad_sphere, 
                  const scalar_t* sphere_index,
                  const int batch, const int channel,
                  const int height, const int width, 
                  const int sphere_size, const int num_votes
                  )
{

  const int num_kernels = channel* num_votes;
  // ***********************************************************************//
  ht2sphere_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_ht, 
                                                          grad_sphere, 
                                                          sphere_index,
                                                          batch, channel, 
                                                          height, width,
                                                          sphere_size, num_votes
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ht2sphere_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

