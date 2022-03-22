#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxWithCrossEntropyKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label, bool use_softmax,
                                   bool numeric_stable_mode, int ignore_index,
                                   int axis, phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss,
                                   phi::DenseTensor* backprop);

}  // namespace phi

