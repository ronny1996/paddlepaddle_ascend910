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
void ReluKernel(const Context& dev_ctx, const phi::DenseTensor& x, phi::DenseTensor* out) {
    dev_ctx.template Alloc<T>(out);
    const auto& runner = NpuOpRunner("Relu", {x}, {*out}, {});

    auto stream = dev_ctx.stream();
    runner.Run(stream);
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& out,
                        const phi::DenseTensor& dout,
                        phi::DenseTensor* dx) {
    auto stream = dev_ctx.stream();
    dev_ctx.template Alloc<T>(dx);
    const auto& runner = NpuOpRunner("ReluGrad", {dout, out}, {*dx}, {});
    runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(relu, Ascend910, ALL_LAYOUT, custom_kernel::ReluKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(
    relu_grad, Ascend910, ALL_LAYOUT, custom_kernel::ReluGradKernel, float, double) {}