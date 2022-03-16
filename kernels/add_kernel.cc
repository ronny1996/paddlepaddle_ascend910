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
#include "npu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
    dev_ctx.template Alloc<T>(out);

    bool direct_compute = false;
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    if (x_dims.size() >= y_dims.size()) {
      direct_compute = x_dims.size() == (y_dims.size() + axis);
    } else {
      direct_compute = y_dims.size() == (x_dims.size() + axis);
    }

    if (direct_compute) {
      const auto& runner = NpuOpRunner("Add", {x, y}, {*out}, {});
      runner.Run(dev_ctx.stream());
    } else {
      phi::DenseTensor transformed_x, transformed_y;
      NpuElementWiseOpBroadcast<T>(dev_ctx, &x, &y, axis, &transformed_x,
                                   &transformed_y);
      const auto& runner =
          NpuOpRunner("Add", {transformed_x, transformed_y}, {*out}, {});
      runner.Run(dev_ctx.stream());
    }
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::AddRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   int axis,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {

    // auto* x = ctx.Input<framework::phi::DenseTensor>("X");
    // auto* y = ctx.Input<framework::phi::DenseTensor>("Y");
    // auto* dout = ctx.Input<framework::phi::DenseTensor>(framework::GradVarName("Out"));
    // auto* dx = ctx.Output<framework::phi::DenseTensor>(framework::GradVarName("X"));
    // auto* dy = ctx.Output<framework::phi::DenseTensor>(framework::GradVarName("Y"));
    // int axis = ctx.Attr<int>("axis");

    axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);
    auto stream = dev_ctx.stream();
    if (dx) {
      // dx->mutable_data<T>(ctx.GetPlace());
      dev_ctx.template Alloc<T>(dx);
      if (dx->dims() != dout.dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        auto src_dims = dx->dims();
        auto dout_dims = dout.dims();

        int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
              (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          } else {
            dst_dims_vec.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes.empty()) {
          phi::DenseTensor tmp;
          tmp.ShareDataWith(*dx);
          tmp.Resize(phi::make_ddim(dst_dims_vec));
          const auto& runner =
              NpuOpRunner("ReduceSumD", {dout}, {tmp},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        TensorCopy(dev_ctx, dout, false, dx);
      }
    }
    if (dy) {
      // dy->mutable_data<T>(ctx.GetPlace());
      dev_ctx.template Alloc<T>(dy);
      if (dy->dims() != dout.dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        auto src_dims = dy->dims();
        auto dout_dims = dout.dims();

        int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
              (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          } else {
            dst_dims_vec.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes.empty()) {
          phi::DenseTensor tmp;
          tmp.ShareDataWith(*dy);
          tmp.Resize(phi::make_ddim(dst_dims_vec));
          const auto& runner =
              NpuOpRunner("ReduceSumD", {dout}, {tmp},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        TensorCopy(dev_ctx, dout, false, dy);
      }
    }
}

} // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_raw,
                          Ascend910,
                          ALL_LAYOUT,
                          custom_kernel::AddRawKernel, int8_t, int32_t, int64_t, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(add,
                          Ascend910,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel, int8_t, int32_t, int64_t, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          Ascend910,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel, int8_t, int32_t, int64_t, float, double) {}