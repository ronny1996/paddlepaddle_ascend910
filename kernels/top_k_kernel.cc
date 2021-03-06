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
void TopkKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                phi::DenseTensor* out,
                phi::DenseTensor* indices) {

    if (axis < 0) {
      axis += x.dims().size();
    }

  int k = k_scalar.to<int>();

    // if (k_tensor != nullptr) {
    //   std::vector<int> v_tmp(1);
    //   paddle::framework::TensorToVector(
    //       k_scalar,
    //       context.template device_context<paddle::platform::NPUDeviceContext>(),
    //       &v_tmp);
    //   k = static_cast<int32_t>(v_tmp[0]);
    // }

    phi::DDim output_dims = x.dims();
    output_dims[axis] = k;

    out->Resize(output_dims);
    indices->Resize(output_dims);

    // out->mutable_data<T>(context.GetPlace());
    dev_ctx.template Alloc<T>(out);
    // indices->mutable_data<int64_t>(context.GetPlace());
    dev_ctx.template Alloc<int64_t>(indices);

    phi::DenseTensor indices_int32(paddle::experimental::DataType::INT32);
    indices_int32.Resize(output_dims);
    // indices_int32.mutable_data<int32_t>(context.GetPlace());
    dev_ctx.template Alloc<int32_t>(&indices_int32);

    auto npu_stream = dev_ctx.stream();

    NpuOpRunner npu_op_runner_topkv2;
    npu_op_runner_topkv2.SetType("TopKV2")
        .AddInput(x)
        .AddInput(std::vector<int32_t>{k})
        .AddOutput(*out)
        .AddOutput(indices_int32)
        .AddAttr("sorted", sorted)
        .AddAttr("dim", axis)
        .AddAttr("largest", largest)
        .Run(npu_stream);

    // Cast 'indices_int32' to 'indices', from INT32 to INT64
    auto dst_dtype =
        ConvertToNpuDtype(indices->type());
    const auto& npu_op_runner_cast =
        NpuOpRunner("Cast", {indices_int32}, {*indices},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    npu_op_runner_cast.Run(npu_stream);
  }

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    top_k, Ascend910, ALL_LAYOUT, custom_kernel::TopkKernel, float, double, int, int64_t) {}