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
            const int num_pts,
            const int num_votes
        )
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(sphere_index.is_contiguous(), "sphere_index tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(sphere_index.type().is_cuda(), "sphere_index must be a CUDA tensor");

    const int batch = input.size(0);
    const int channel = input.size(1);
    const int num_elements = sphere_index.size(-1);
    AT_ASSERTM(num_elements == 2, "num_elements != num_elements: (%d).", num_elements);
    

    auto output = at::zeros({batch, channel, num_pts}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "ht2sphere_cuda_forward", ([&] {
        ht2sphere_cuda_forward(at::cuda::getCurrentCUDAStream(),
                    input.data<scalar_t>() ,
                    output.data<scalar_t>(),
                    sphere_index.data<scalar_t>(),
                    batch, channel,
                    height, width, 
                    num_pts, num_votes
                    );

    }));
    // std::cout <<"output" <<output.sum() << std::endl;
    // printf("output", output.sum());
    // output = output.contiguous().view({batch, channel, num_pts});
    return output;
}



at::Tensor
sphere_cuda_backward(
            const at::Tensor &grad_output, 
            const at::Tensor &sphere_index,
            const int height,
            const int width,
            const int num_pts,
            const int num_votes
            )
{

    AT_ASSERTM(sphere_index.is_contiguous(), "sphere_index tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(sphere_index.type().is_cuda(), "sphere_index must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");

    // grad_output: [batch, channel, sphere_size]
    const int batch = grad_output.size(0);
    const int channel = grad_output.size(1);

    AT_ASSERTM(num_pts == grad_output.size(2),
        "grad_out shape and num_pts : (%d vs %d).", num_pts, grad_output.size(2));
    
    auto grad_input = at::zeros({batch, channel, height, width}, grad_output.options());
    
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "ht2sphere_cuda_backward", ([&] {
        ht2sphere_cuda_backward(at::cuda::getCurrentCUDAStream(),
                    grad_input.data<scalar_t>(),
                    grad_output.data<scalar_t>(),
                    sphere_index.data<scalar_t>(),
                    batch, channel,
                    height, width, 
                    num_pts, num_votes
                );

    }));

    return grad_input; 
}
