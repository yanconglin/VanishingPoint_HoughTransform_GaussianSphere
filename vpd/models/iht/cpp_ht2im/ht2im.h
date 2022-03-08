#pragma once
#include <torch/extension.h>

// std::vector<at::Tensor>
at::Tensor
ht2im_forward(
            const at::Tensor &input,
            const at::Tensor &ht_index,
            const int height,
            const int width,
            const int ht_height,
            const int ht_width
            );

at::Tensor
ht2im_backward(const at::Tensor &grad_output,
                const at::Tensor &ht_index,
                const int height,
                const int width,
                const int ht_height,
                const int ht_width
                );
