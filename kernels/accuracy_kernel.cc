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
void AccuracyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& indices,
                       const phi::DenseTensor& label,
                       phi::DenseTensor* accuracy,
                       phi::DenseTensor* correct,
                       phi::DenseTensor* total) {

    auto stream = dev_ctx.stream();

    int num_samples = out.dims()[0];
    if (num_samples == 0) {
      return;
    }

    // cast `indices` or `label` if their type is not consistent
    phi::DenseTensor cast_indices(paddle::experimental::DataType::INT32);
    phi::DenseTensor cast_label(paddle::experimental::DataType::INT32);
    if (indices.dtype() != label.dtype()) {
      auto dst_dtype = ConvertToNpuDtype(paddle::experimental::DataType::INT32);
      if (indices.dtype() != paddle::experimental::DataType::INT32) {
        cast_indices.Resize(indices.dims());
        dev_ctx.template Alloc<int>(&cast_indices);
        //cast_indices.mutable_data<int>(ctx.GetPlace());
        const auto& runner_cast_indices =
            NpuOpRunner("Cast", {indices}, {cast_indices},
                        {{"dst_type", static_cast<int>(dst_dtype)}});
        runner_cast_indices.Run(stream);
      } else {
        cast_indices.ShareDataWith(indices);
      }
      if (label.dtype() != paddle::experimental::DataType::INT32) {
        cast_label.Resize(label.dims());
        //cast_label.mutable_data<int>(ctx.GetPlace());
        dev_ctx.template Alloc<int>(&cast_label);
        const auto& runner_cast_label =
            NpuOpRunner("Cast", {label}, {cast_label},
                        {{"dst_type", static_cast<int>(dst_dtype)}});
        runner_cast_label.Run(stream);
      } else {
        cast_label.ShareDataWith(label);
      }
    } else {
      cast_indices.ShareDataWith(indices);
      cast_label.ShareDataWith(label);
    }

    // equal
    phi::DenseTensor tmp_equal(paddle::experimental::DataType::BOOL);
    tmp_equal.Resize(out.dims());
    // tmp_equal.mutable_data<bool>(ctx.GetPlace());
    dev_ctx.template Alloc<bool>(&tmp_equal);
    const auto& runner_equal =
        NpuOpRunner("Equal", {cast_indices, cast_label}, {tmp_equal}, {});
    runner_equal.Run(stream);

    // cast equal
    phi::DenseTensor tmp_equal_cast(paddle::experimental::DataType::FLOAT32);
    tmp_equal_cast.Resize(out.dims());
    dev_ctx.template Alloc<float>(&tmp_equal_cast);
    // tmp_equal_cast.mutable_data<float>(ctx.GetPlace());
    const auto& runner_cast_equal = NpuOpRunner(
        "Cast", {tmp_equal}, {tmp_equal_cast},
        {{"dst_type",
          static_cast<int>(ConvertToNpuDtype(tmp_equal_cast.dtype()))}});
    runner_cast_equal.Run(stream);

    // [correct]
    // reduce_max
    phi::DenseTensor tmp_correct_max(paddle::experimental::DataType::FLOAT32);
    tmp_correct_max.Resize(phi::make_ddim({num_samples}));
    // tmp_correct_max.mutable_data<float>(ctx.GetPlace());
    dev_ctx.template Alloc<float>(&tmp_correct_max);
    const auto& runner_reduce_max =
        NpuOpRunner("ReduceMaxD", {tmp_equal_cast}, {tmp_correct_max},
                    {{"axes", std::vector<int>{1}}, {"keep_dims", false}});
    runner_reduce_max.Run(stream);

    // reduce_sum
    phi::DenseTensor tmp_correct(paddle::experimental::DataType::FLOAT32);
    tmp_correct.Resize(correct->dims());
    // tmp_correct.mutable_data<float>(ctx.GetPlace());
    dev_ctx.template Alloc<float>(&tmp_correct);
    const auto& runner_reduce_sum =
        NpuOpRunner("ReduceSumD", {tmp_correct_max}, {tmp_correct},
                    {{"axes", std::vector<int>{0}}, {"keep_dims", false}});
    runner_reduce_sum.Run(stream);

    // cast to int
    // correct->mutable_data<int>(ctx.GetPlace());
    dev_ctx.template Alloc<int>(correct);
    const auto& runner_cast_correct = NpuOpRunner(
        "Cast", {tmp_correct}, {*correct},
        {{"dst_type", static_cast<int>(ConvertToNpuDtype(correct->dtype()))}});
    runner_cast_correct.Run(stream);

    // [total]
    // total->mutable_data<int>(ctx.GetPlace());
    dev_ctx.template Alloc<int>(total);
    FillNpuTensorWithConstant<int>(total, dev_ctx, static_cast<int>(num_samples));

    // use `total` of type `float32` for calculating accuracy
    phi::DenseTensor tmp_total(paddle::experimental::DataType::FLOAT32);
    tmp_total.Resize(total->dims());
    // tmp_total.mutable_data<float>(ctx.GetPlace());
    dev_ctx.template Alloc<float>(&tmp_total);
    FillNpuTensorWithConstant<float>(&tmp_total, dev_ctx,
                                     static_cast<float>(num_samples));

    // [accuracy]
    // accuracy->mutable_data<float>(ctx.GetPlace());
    dev_ctx.template Alloc<float>(accuracy);
    const auto& runner_accuracy =
        NpuOpRunner("Div", {tmp_correct, tmp_total}, {*accuracy}, {});
    runner_accuracy.Run(stream);
  }

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    accuracy, Ascend910, ALL_LAYOUT, custom_kernel::AccuracyRawKernel, float, phi::dtype::float16, int, int64_t) {}