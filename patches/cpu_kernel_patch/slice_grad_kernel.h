#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void SliceGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                     const phi::DenseTensor& out_grad,
                     const phi::ScalarArray& axes_array,
                     const phi::ScalarArray& starts_array,
                     const phi::ScalarArray& ends_array,
                     phi::DenseTensor* x_grad);

}  // namespace phi

