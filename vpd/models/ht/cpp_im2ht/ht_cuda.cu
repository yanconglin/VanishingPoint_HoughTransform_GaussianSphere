#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "im2ht_cuda.cuh"


at::Tensor
ht_cuda_forward(
            const at::Tensor &input,
            const at::Tensor &ht_index,
            const int height,
            const int width,
            const int ht_height,
            const int ht_width
        )
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(ht_index.is_contiguous(), "ht_index tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(ht_index.type().is_cuda(), "ht_index must be a CUDA tensor");

    const int batch = input.size(0);
    const int channel = input.size(1);
    const int height_ = input.size(2);
    const int width_ = input.size(3);

    const int num_votes = ht_index.size(0);
    const int num_elements = ht_index.size(1);
    AT_ASSERTM(num_elements == 3,
        "ht_index shape != [num_votes x 3]: (%d x %d).", num_votes, num_elements);
    
    AT_ASSERTM(height_ == height && width_== width,
        " image shape and given shape params do not match: (%d x %d vs %d x %d).", height_, width_, height, width);

    // AT_ASSERTM(ht_h <= ht_index.max().item(), 
    //     "ht_index larger than ht_h (%d vs %d).", ht_index.max().item(), ht_h);

    // output: [batch, channel,  ht_height, ht_width]
    auto output = at::zeros({batch * channel * ht_height * ht_width}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "im2ht_cuda_forward", ([&] {
        im2ht_cuda_forward(at::cuda::getCurrentCUDAStream(),
                    input.data<scalar_t>() ,
                    output.data<scalar_t>(),
                    ht_index.data<scalar_t>(),
                    batch, channel,
                    height, width, 
                    ht_height, ht_width, 
                    num_votes
                    );

    }));
    // std::cout <<"output" <<output.sum() << std::endl;
    // printf("output", output.sum());
    output = output.contiguous().view({batch,channel, ht_height, ht_width});
    return output;
}



at::Tensor
ht_cuda_backward(
            const at::Tensor &grad_output, 
            const at::Tensor &ht_index,
            const int height,
            const int width,
            const int ht_height,
            const int ht_width
            )
{

    AT_ASSERTM(ht_index.is_contiguous(), "ht_index tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(ht_index.type().is_cuda(), "ht_index must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");

    // grad_output: [batch, channel, ht_h, ht_w]
    const int batch = grad_output.size(0);
    const int channel = grad_output.size(1);
    const int ht_height_ = grad_output.size(2);
    const int ht_width_ = grad_output.size(3);

    AT_ASSERTM(ht_height_ == ht_height && ht_width_ == ht_width,
        "given grad_out shape and ht_index shape do not match: (%d x %d vs %d).", ht_height_, ht_width_, ht_height, ht_width);

    const int num_votes = ht_index.size(0);
    const int num_elements = ht_index.size(1);
    AT_ASSERTM(num_elements == 3,
        "ht_index shape != [num_votes x 3]: (%d x %d).", num_votes, num_elements);
    
    auto grad_input = at::zeros({batch, channel, height, width}, grad_output.options());
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "im2ht_cuda_backward", ([&] {
        im2ht_cuda_backward(at::cuda::getCurrentCUDAStream(),
                    grad_input.data<scalar_t>(),
                    grad_output.data<scalar_t>(),
                    ht_index.data<scalar_t>(),
                    batch, channel,
                    height, width, 
                    ht_height, ht_width,
                    num_votes
                );

    }));

    return grad_input; 
}
