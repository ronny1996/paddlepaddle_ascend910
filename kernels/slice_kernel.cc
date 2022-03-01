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

#include "npu_op_runner.h"

/*
namespace custom_kernel {

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const std::vector<phi::DenseTensor>& starts_tensor_list,
                  const std::vector<phi::DenseTensor>& starts_tensor_list,
                  const std::vector<int>& axes_int,
                  const std::vector<int>& starts_int,
                  const std::vector<int>& ends_int,
                  const std::vector<int>& decrease_axis,
                  const std::vector<int>& infer_flags,
                  phi::DenseTensor* out) {

    std::vector<int> axes(axes_int.begin(), axes_int.end());
    std::vector<int> starts(starts_int.begin(), starts_int.end());
    std::vector<int> ends(ends_int.begin(), ends_int.end());

    const auto& in_dims = input->dims();
    // Get the accurate attribute value of starts and ends
    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts = GetDataFromTensor<int>(ctx.Input<Tensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int>(ctx.Input<Tensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int>(ends_tensor_list);
    }

    PADDLE_ENFORCE_EQ(
        starts.size(), axes.size(),
        phi::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(), axes.size(),
        phi::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));

    if (ctx.HasInput("StartsTensor") || ctx.HasInput("EndsTensor") ||
        starts_tensor_list.size() > 0 || ends_tensor_list.size() > 0) {
      // Infer output dims
      auto out_dims = out->dims();
      auto slice_dims = out_dims;
      for (size_t i = 0; i < axes.size(); ++i) {
        // when start == -1 && end == start+1
        if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
          auto ret =
              std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
          if (ret != decrease_axis.end()) {
            ends[i] = in_dims[axes[i]];
          }
        }
      }

      CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
      slice_dims =
          GetSliceDims<int>(in_dims, axes, starts, ends, nullptr, nullptr);
      out_dims = GetDecreasedDims(slice_dims, decrease_axis);

      out->Resize(out_dims);
    }

    out->mutable_data<T>(ctx.GetPlace());

    std::vector<int> offsets(in_dims.size());
    std::vector<int> size(in_dims.size());

    UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    const auto& runner = NpuOpRunner("SliceD", {*input}, {*out},
                                     {{"offsets", offsets}, {"size", size}});
    runner.Run(stream);
  }

} // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice,
                   Ascend910,
                   ALL_LAYOUT,
                   custom_kernel::SliceKernel,
                   bool,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double) {}
*/