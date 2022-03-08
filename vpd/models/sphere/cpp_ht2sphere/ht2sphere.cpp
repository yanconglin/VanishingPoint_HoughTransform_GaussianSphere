#include "ht2sphere.h"
#include "sphere_cuda.h"

// std::vector<at::Tensor>
at::Tensor
ht2sphere_forward(
            const at::Tensor &input,
            const at::Tensor &sphere_index,
            const int height,
            const int width,
            const int sphere_size
            )
{
    if (input.type().is_cuda())
    {
        return sphere_cuda_forward(input, 
                                sphere_index,
                                height,
                                width,
                                sphere_size  
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor
ht2sphere_backward(const at::Tensor &grad_output,
                const at::Tensor &sphere_index,
                const int height,
                const int width,
                const int sphere_size
                )
{
    if (grad_output.type().is_cuda())
    {
        return sphere_cuda_backward(
                                grad_output, 
                                sphere_index,
                                height,
                                width,
                                sphere_size
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ht2sphere_forward", &ht2sphere_forward, "Forward pass of sphere");
    m.def("ht2sphere_backward", &ht2sphere_backward, "Backward pass of sphere");
}
