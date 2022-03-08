#pragma once
#include <torch/extension.h>


// std::vector<at::Tensor>
at::Tensor
sphere_cuda_forward(
                const at::Tensor &input,
                const at::Tensor &sphere_index,
                const int height,
                const int width,
                const int sphere_size
                );

at::Tensor
sphere_cuda_backward(
                const at::Tensor &grad_output, 
                const at::Tensor &sphere_index,
                const int height,
                const int width,
                const int sphere_size
                );

