#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                 const phi::ScalarArray& axes_array,
                 const phi::ScalarArray& starts_array,
                 const phi::ScalarArray& ends_array, phi::DenseTensor* out);

}  // namespace phi

