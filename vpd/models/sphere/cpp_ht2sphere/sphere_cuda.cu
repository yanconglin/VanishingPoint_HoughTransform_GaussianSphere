#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "ht2sphere_cuda.cuh"


at::Tensor
sphere_cuda_forward(
            const at::Tensor &input,
            const at::Tensor &sphere_index,
            const int height,
            const int width,
            const int sphere_size
        )
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(sphere_index.is_contiguous(), "sphere_index tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(sphere_index.type().is_cuda(), "sphere_index must be a CUDA tensor");

    const int batch = input.size(0);
    const int channel = input.size(1);
    const int height_ = input.size(2);
    const int width_ = input.size(3);

    AT_ASSERTM(height==height_ && width==width_,
        "(height!=height_ && width!=width_): (%d x %d) vs (%d x %d).", height, width, height_, width_);
    

    const int num_votes = sphere_index.size(0);
    const int num_elements = sphere_index.size(1);
    AT_ASSERTM(num_elements == 3,
        "sphere_index shape != [num_votes x 3]: (%d x %d).", num_votes, num_elements);
    

    // output: [batch, channel, ht_h, ht_w]
    auto output = at::zeros({batch * channel * sphere_size}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "ht2sphere_cuda_forward", ([&] {
        ht2sphere_cuda_forward(at::cuda::getCurrentCUDAStream(),
                    input.data<scalar_t>() ,
                    output.data<scalar_t>(),
                    sphere_index.data<scalar_t>(),
                    batch, channel,
                    height, width, 
                    sphere_size, num_votes
                    );

    }));
    // std::cout <<"output" <<output.sum() << std::endl;
    // printf("output", output.sum());
    output = output.contiguous().view({batch, channel, sphere_size});
    return output;
}



at::Tensor
sphere_cuda_backward(
            const at::Tensor &grad_output, 
            const at::Tensor &sphere_index,
            const int height,
            const int width,
            const int sphere_size
            )
{

    AT_ASSERTM(sphere_index.is_contiguous(), "sphere_index tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(sphere_index.type().is_cuda(), "sphere_index must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");

    // grad_output: [batch, channel, sphere_size]
    const int batch = grad_output.size(0);
    const int channel = grad_output.size(1);
    const int sphere_size_grad = grad_output.size(2);

    const int num_votes = sphere_index.size(0);
    const int num_elements = sphere_index.size(1);

    AT_ASSERTM(num_elements == 3,
        "sphere_index shape != [num_votes x 3]: (%d x %d).", num_votes, num_elements);

    AT_ASSERTM(sphere_size == sphere_size_grad,
        "grad_out shape and sphere_size : (%d x %d vs %d).", sphere_size_grad, sphere_size);
    
    auto grad_input = at::zeros({batch, channel, height, width}, grad_output.options());
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "ht2sphere_cuda_backward", ([&] {
        ht2sphere_cuda_backward(at::cuda::getCurrentCUDAStream(),
                    grad_input.data<scalar_t>(),
                    grad_output.data<scalar_t>(),
                    sphere_index.data<scalar_t>(),
                    batch, channel,
                    height, width, 
                    sphere_size, num_votes
                );

    }));

    return grad_input; 
}
