
#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype, bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void SoftmaxWithCrossEntropyKernel(const Context& dev_ctx,
                   const phi::DenseTensor& logits,
                   const phi::DenseTensor& labels,
                   bool soft_label,
                   bool use_softmax,
                   bool numeric_stable_mode,
                   int ignore_index,
                   int axis,
                   phi::DenseTensor* softmax,
                   phi::DenseTensor* loss,
                   phi::DenseTensor* backprop);

template <typename T, typename Context>
void SoftmaxWithCrossEntropyGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& labels,
                   const phi::DenseTensor& softmax,
                   const phi::DenseTensor& backprop,
                   const phi::DenseTensor& loss_grad,
                   bool soft_label,
                   bool use_softmax,
                   bool numeric_stable_mode,
                   int ignore_index,
                   int axis,
                   phi::DenseTensor* logits_grad);

}  // namespace phi
