#pragma once
#include <torch/extension.h>


// std::vector<at::Tensor>
at::Tensor
sphere_cuda_forward(
                const at::Tensor &input,
                const at::Tensor &sphere_index,
                const int height,
                const int width,
                const int num_pts,
                const int num_votes
                );

at::Tensor
sphere_cuda_backward(
                const at::Tensor &grad_output, 
                const at::Tensor &sphere_index,
                const int height,
                const int width,
                const int num_pts,
                const int num_votes
                );

