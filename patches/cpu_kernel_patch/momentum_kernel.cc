#include "paddle/phi/kernels/momentum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MomentKernel(const Context& dev_ctx, const phi::DenseTensor& param,
                  const phi::DenseTensor& grad,
                  const phi::DenseTensor& velocity,
                  const phi::DenseTensor& learning_rate, float mu_f,
                  bool use_nesterov, const std::string& regularization_method,
                  float regularization_coeff, bool multi_precision,
                  float rescale_grad, phi::DenseTensor* param_out,
                  phi::DenseTensor* velocity_out) {}

}  // namespace phi

PD_REGISTER_KERNEL(momentum, CPU, ALL_LAYOUT, phi::MomentKernel, float, double) {}

