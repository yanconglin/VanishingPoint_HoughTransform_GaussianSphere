#include "im2ht.h"
#include "ht_cuda.h"

// std::vector<at::Tensor>
at::Tensor
im2ht_forward(
            const at::Tensor &input,
            const at::Tensor &ht_index,
            const int height,
            const int width,
            const int ht_height,
            const int ht_width
            )
{
    if (input.type().is_cuda()  && ht_index.type().is_cuda())
    {
        return ht_cuda_forward(input, 
                                ht_index,
                                height, width,
                                ht_height, ht_width
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor
im2ht_backward(const at::Tensor &grad_output,
                const at::Tensor &ht_index,
                const int height,
                const int width,
                const int ht_height,
                const int ht_width
                )
{
    if (grad_output.type().is_cuda() && ht_index.type().is_cuda())
    {
        return ht_cuda_backward(
                                grad_output, 
                                ht_index,
                                height, width,
                                ht_height, ht_width
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("im2ht_forward", &im2ht_forward, "Forward pass of ht");
    m.def("im2ht_backward", &im2ht_backward, "Backward pass of ht");
}
