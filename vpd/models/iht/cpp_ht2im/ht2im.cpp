#include "ht2im.h"
#include "iht_cuda.h"


// std::vector<at::Tensor>
at::Tensor
ht2im_forward(
            const at::Tensor &input_ht,
            const at::Tensor &ht_index,
            const int height,
            const int width,
            const int ht_height,
            const int ht_width
            )
{
    if (input_ht.type().is_cuda())
    {
        return iht_cuda_forward(input_ht, 
                                ht_index,
                                height,
                                width,
                                ht_height,
                                ht_width
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor
ht2im_backward(const at::Tensor &grad_output,
                const at::Tensor &ht_index,
                const int height,
                const int width,
                const int ht_height,
                const int ht_width
                )
{
    if (grad_output.type().is_cuda())
    {
        return iht_cuda_backward(
                                grad_output, 
                                ht_index,
                                height,
                                width,
                                ht_height,
                                ht_width
                                );
    }
    AT_ERROR("Not implemented on the CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ht2im_forward", &ht2im_forward, "Forward pass of iht");
    m.def("ht2im_backward", &ht2im_backward, "Backward pass of iht");
}
