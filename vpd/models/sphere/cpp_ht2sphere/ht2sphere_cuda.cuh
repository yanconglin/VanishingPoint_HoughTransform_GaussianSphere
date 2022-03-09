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
                                  const int num_pts, const int num_votes
                                )
{
  // launch num_pts*num_votes*channel  cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // int b = index / num_pts / num_votes / channel % batch; // b
    // int h = index / width / num_pts /num_votes % height; // h
    // int w = index / num_pts /num_votes % width; // w
    int s = index / num_votes / channel % num_pts; // s
    int v = index / channel % num_votes;  // v
    int c = index  % channel; // c

    for (int b=0; b<batch; b++)
    {
      // sphere_index: [B, num_pts, num_votes, 2]
      int hw = sphere_index[b*num_pts*num_votes*2 + s*num_votes*2 + v*2 + 0];
      scalar_t sphere_vote = sphere_index[b*num_pts*num_votes*2 + s*num_votes*2 + v*2 + 1];

      // output: [batch, channel, num_pts]
      if (sphere_vote>0.0)
      {
        int offset_ht = b * (channel * height * width) + c * (height * width) + hw;
        int offset_sphere =  b * (channel * num_pts) + c * num_pts + s; // offset in the sphere
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
                                    const int num_pts, const int num_votes
                                  )
{
  // launch num_pts*num_votes*channel  cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // grad_sphere: [batch, channel, sphere_size]
    // grad_ht: [batch, channel, height, width]
    // int b = index / num_pts / num_votes / channel % batch; // b
    // int h = index / width / num_pts /num_votes % height; // h
    // int w = index / num_pts /num_votes % width; // w
    int s = index / num_votes / channel % num_pts; // s
    int v = index / channel % num_votes;  // v
    int c = index  % channel; // c

    for (int b=0; b<batch; b++)
    {
      // sphere_index: [B, num_pts, num_votes, 2]
      int hw = sphere_index[b*num_pts*num_votes*2 + s*num_votes*2 + v*2 + 0];
      scalar_t sphere_vote = sphere_index[b*num_pts*num_votes*2 + s*num_votes*2 + v*2 + 1];
      if (sphere_vote>0.0)
      {
        int offset_ht = b * (channel * height * width) + c * (height * width) + hw;
        int offset_sphere =  b * (channel * num_pts) + c * num_pts + s; // offset in the sphere
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
                  const int num_pts, const int num_votes
                ) 
{
  const int num_kernels = num_pts*num_votes*num_pts;
  ht2sphere_cuda_forward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(num_kernels, 
                                                          data_ht, 
                                                          data_sphere,
                                                          sphere_index, 
                                                          batch, channel, 
                                                          height, width, 
                                                          num_pts, num_votes
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
                  const int num_pts, const int num_votes
                  )
{

  const int num_kernels = num_pts*num_votes*channel;
  // ***********************************************************************//
  ht2sphere_cuda_backward_kernel<scalar_t>
  <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,0, stream>>>(num_kernels, 
                                                          grad_ht, 
                                                          grad_sphere, 
                                                          sphere_index,
                                                          batch, channel, 
                                                          height, width,
                                                          num_pts, num_votes
                                                          );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ht2sphere_gpu_kernel: %s\n", cudaGetErrorString(err));
  }

}

