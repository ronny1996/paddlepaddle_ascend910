#include "paddle/phi/kernels/slice_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                 const phi::ScalarArray& axes_array,
                 const phi::ScalarArray& starts_array,
                 const phi::ScalarArray& ends_array, phi::DenseTensor* out) {}

}  // namespace phi

PD_REGISTER_KERNEL(slice, CPU, ALL_LAYOUT, phi::SliceKernel, float, double,
                   bool) {}

