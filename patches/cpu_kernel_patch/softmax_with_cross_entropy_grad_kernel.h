#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxWithCrossEntropyGradKernel(
    const Context& dev_ctx, const phi::DenseTensor& labels,
    const phi::DenseTensor& softmax, const phi::DenseTensor& backprop,
    const phi::DenseTensor& loss_grad, bool soft_label, bool use_softmax,
    bool numeric_stable_mode, int ignore_index, int axis,
    phi::DenseTensor* logits_grad);

}  // namespace phi

