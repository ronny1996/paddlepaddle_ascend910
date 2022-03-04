// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "npu_funcs.h"
#include "npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void MomentKernel(const Context& dev_ctx, const phi::DenseTensor& param,
                  const phi::DenseTensor& grad,
                  const phi::DenseTensor& velocity,
                  const phi::DenseTensor& learning_rate, float mu_f,
                  bool use_nesterov, const std::string& regularization_method,
                  float regularization_coeff, bool multi_precision,
                  float rescale_grad, phi::DenseTensor* param_out,
                  phi::DenseTensor* velocity_out) {
  // auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();

  auto mu = static_cast<T>(mu_f);

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(velocity_out);

  // if (grad_var->IsType<LoDTensor>()) {
  phi::DenseTensor mu_tensor;
  mu_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&mu_tensor);
  FillNpuTensorWithConstant<T>(&mu_tensor, dev_ctx, mu);

  phi::DenseTensor regularized_grad;
  if (regularization_method == "l2_decay") {
    regularized_grad.Resize({1});
    dev_ctx.template Alloc<T>(&regularized_grad);

    const auto& runner1 = NpuOpRunner("Muls", {param}, {regularized_grad},
                                      {{"value", regularization_coeff}});
    runner1.Run(dev_ctx.stream());
    const auto& runner2 =
        NpuOpRunner("Add", {regularized_grad, grad}, {regularized_grad}, {});
    runner2.Run(dev_ctx.stream());
  } else {
    regularized_grad.ShareDataWith(grad);
  }
  TensorCopy(dev_ctx, param, false, param_out);
  TensorCopy(dev_ctx, velocity, false, velocity_out);
  // NOTE: ApplyMomentum will change the input
  const auto& runner = NpuOpRunner(
      "ApplyMomentum",
      {*param_out, *velocity_out, learning_rate, regularized_grad, mu_tensor},
      {*param_out}, {{"use_nesterov", use_nesterov}});
  runner.Run(dev_ctx.stream());
  // } else if (grad_var->IsType<phi::SelectedRows>()) {
  //   PADDLE_ENFORCE_EQ(false, true, platform::errors::PermissionDenied(
  //                                      "Unsupport SparseMomentum"));
  // } else {
  //   PADDLE_ENFORCE_EQ(false, true,
  //                     platform::errors::PermissionDenied(
  //                         "Unsupported Variable Type of Grad "
  //                         "in MomentumOp. Excepted LodTensor "
  //                         "or SelectedRows, But received [%s]",
  //                         paddle::ToTypeName(grad_var->Type())));
  // }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(moment, Ascend910, ALL_LAYOUT,
                          custom_kernel::MomentKernel, phi::dtype::float16,
                          float, double) {}