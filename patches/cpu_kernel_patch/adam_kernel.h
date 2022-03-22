#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void AdamKernel(const Context& dev_ctx, const phi::DenseTensor& param,
                const phi::DenseTensor& grad,
                const phi::DenseTensor& learning_rate,
                const phi::DenseTensor& moment1,
                const phi::DenseTensor& moment2,
                const phi::DenseTensor& beta1_pow_in,
                const phi::DenseTensor& beta2_pow_in,
                /*
                                const phi::DenseTensor& beta1_tensor,
                                const phi::DenseTensor& beta2_tensor,
                                const phi::DenseTensor& epsilon_tensor,
                                const phi::DenseTensor& master_param,
                                const phi::DenseTensor& skip_update,
                */
                float beta1_f, float beta2_f, float epsilon_f, bool lazy_mode,
                int64_t min_row_size_to_use_multithread, bool multi_precision,
                bool use_global_beta_pow, phi::DenseTensor* param_out,
                phi::DenseTensor* moment1_out, phi::DenseTensor* moment2_out,
                phi::DenseTensor* beta1_pow_out, phi::DenseTensor* beta2_pow_out
                /* phi::DenseTensor* master_param_out */
                );
}  // namespace phi

