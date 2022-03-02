
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

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
void MaxRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype, bool keep_dim,
               phi::DenseTensor* out);

}  // namespace phi
