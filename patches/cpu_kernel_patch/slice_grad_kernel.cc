#include "paddle/phi/kernels/slice_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SliceGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                     const phi::DenseTensor& out_grad,
                     const phi::ScalarArray& axes_array,
                     const phi::ScalarArray& starts_array,
                     const phi::ScalarArray& ends_array,
                     phi::DenseTensor* x_grad) {}

}  // namespace phi

PD_REGISTER_KERNEL(slice_grad, CPU, ALL_LAYOUT, phi::SliceGradKernel, float,
                   double, bool) {}

