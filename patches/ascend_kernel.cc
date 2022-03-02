#include "paddle/phi/kernels/ascend_kernel.h"

namespace phi {

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype, bool keep_dim,
               phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype, bool keep_dim,
               phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                    const std::vector<int64_t>& dim, bool keep_dim,
                    phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MeanRawGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                       const phi::DenseTensor& out_grad,
                       const std::vector<int64_t>& dim, bool keep_dim,
                       bool reduce_all, phi::DenseTensor* x_grad) {}

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                 const phi::ScalarArray& axes_array,
                 const phi::ScalarArray& starts_array,
                 const phi::ScalarArray& ends_array, phi::DenseTensor* out) {}

template <typename T, typename Context>
void SGDKernel(const Context& dev_ctx, const phi::DenseTensor& param_var,
               const phi::DenseTensor& learning_rate,
               const phi::DenseTensor& grad_var, phi::DenseTensor* param_out) {}

}  // namespace phi

PD_REGISTER_KERNEL(max_raw, CPU, ALL_LAYOUT, phi::MaxRawKernel, float, double,
                   bool) {}

PD_REGISTER_KERNEL(min_raw, CPU, ALL_LAYOUT, phi::MinRawKernel, float, double,
                   bool) {}

PD_REGISTER_KERNEL(max, CPU, ALL_LAYOUT, phi::MaxKernel, float, double, bool) {}

PD_REGISTER_KERNEL(min, CPU, ALL_LAYOUT, phi::MinKernel, float, double, bool) {}

PD_REGISTER_KERNEL(mean_raw_grad, CPU, ALL_LAYOUT, phi::MeanRawGradKernel,
                   float, double, bool) {}

PD_REGISTER_KERNEL(mean_grad, CPU, ALL_LAYOUT, phi::MeanGradKernel, float,
                   double, bool) {}

PD_REGISTER_KERNEL(slice, CPU, ALL_LAYOUT, phi::SliceKernel, float, double,
                   bool) {}

PD_REGISTER_KERNEL(sgd, CPU, ALL_LAYOUT, phi::SGDKernel, float, double, bool) {}
