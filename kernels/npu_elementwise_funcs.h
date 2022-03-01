/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "npu_op_runner.h"
#include "npu_enforce.h"

namespace custom_kernel {

template <typename T, typename Context>
void NpuBroadcast(const Context& dev_ctx, const phi::DenseTensor* src,
                  int axis, const phi::DDim& dst_dims,
                  phi::DenseTensor* transformed_src) {
  auto stream = dev_ctx.stream();

  // 1. expand the axis with dim 1
  auto src_dims = src->dims();
  phi::DenseTensor tmp_src;
  tmp_src.ShareDataWith(*src);
  tmp_src.Resize(src_dims);
  for (int i = 0; i < src_dims.size(); ++i) {
    if (src_dims[i] == 1 && dst_dims[i + axis] > 1) {
      phi::DenseTensor tmp_tensor;
      auto tmp_tensor_dims = tmp_src.dims();
      tmp_tensor_dims[i] = dst_dims[i + axis];
      tmp_tensor.Resize(tmp_tensor_dims);
      dev_ctx.template Alloc<T>(&tmp_tensor);
      const auto& runner =
          NpuOpRunner("TileWithAxis", {tmp_src}, {tmp_tensor},
                      {{"axis", static_cast<int64_t>(i)},
                       {"tiles", static_cast<int64_t>(dst_dims[i + axis])}});
      runner.Run(stream);
      tmp_src.ShareDataWith(tmp_tensor);
      tmp_src.Resize(tmp_tensor_dims);
    }
  }

  // 2.expand the ahead axis
  auto prev = phi::product(phi::slice_ddim(dst_dims, 0, axis));
  if (prev > 1) {
    phi::DenseTensor tmp_tensor;
    auto tmp_tensor_dims = phi::slice_ddim(dst_dims, 0, axis + src_dims.size());
    tmp_tensor.Resize(tmp_tensor_dims);
    dev_ctx.template Alloc<T>(&tmp_tensor);
    const auto& runner =
        NpuOpRunner("ExpandD", {tmp_src}, {tmp_tensor},
                    {{"shape", phi::vectorize<int64_t>(tmp_tensor_dims)}});
    runner.Run(stream);
    tmp_src.ShareDataWith(tmp_tensor);
    tmp_src.Resize(tmp_tensor_dims);
  } else {
    tmp_src.Resize(phi::slice_ddim(dst_dims, 0, axis + src_dims.size()));
  }

  // 3.expand the tail axis
  auto post = phi::product(
      phi::slice_ddim(dst_dims, axis + src_dims.size(), dst_dims.size()));
  if (post > 1) {
    auto src_dims_vec = phi::vectorize<int>(tmp_src.dims());
    src_dims_vec.push_back(1);
    tmp_src.Resize(phi::make_ddim(src_dims_vec));

    phi::DenseTensor tmp_tensor;
    tmp_tensor.Resize(dst_dims);
    dev_ctx.template Alloc<T>(&tmp_tensor);
    const auto& runner =
        NpuOpRunner("TileWithAxis", {tmp_src}, {tmp_tensor},
                    {{"axis", static_cast<int64_t>(axis + src_dims.size())},
                     {"tiles", static_cast<int64_t>(post)}});
    runner.Run(stream);
    tmp_src.ShareDataWith(tmp_tensor);
  }
  tmp_src.Resize(dst_dims);
  //TODO:
  //framework::TensorCopy(tmp_src, dev_ctx.GetPlace(), transformed_src);
}

template <typename T, typename Context>
void NpuElementWiseOpBroadcast(const Context& dev_ctx,
                               const phi::DenseTensor* x, const phi::DenseTensor* y, int axis,
                               phi::DenseTensor* transformed_x, phi::DenseTensor* transformed_y) {
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  std::vector<int> dst_dims_vec = phi::vectorize<int>(x_dims);

  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
    dst_dims_vec = phi::vectorize<int>(y_dims);
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  int x_axis = is_xsize_larger ? 0 : axis;
  int y_axis = is_xsize_larger ? axis : 0;

  PADDLE_ENFORCE_GE(
      axis, 0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    phi::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));

  for (int i = 0; i < x_dims.size(); ++i) {
    dst_dims_vec[i + x_axis] =
        std::max(dst_dims_vec[i + x_axis], static_cast<int>(x_dims[i]));
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    dst_dims_vec[i + y_axis] =
        std::max(dst_dims_vec[i + y_axis], static_cast<int>(y_dims[i]));
  }

  auto dst_dims = phi::make_ddim(dst_dims_vec);
  NpuBroadcast<T>(dev_ctx, x, x_axis, dst_dims, transformed_x);
  NpuBroadcast<T>(dev_ctx, y, y_axis, dst_dims, transformed_y);
}

}  // namespace custom_kernel 

