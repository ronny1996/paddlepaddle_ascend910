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

#include <memory>
#include "npu_op_runner.h"
#include "npu_enforce.h"

namespace custom_kernel {

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const phi::DeviceContext& dev_ctx, phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  phi::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<T>(dst));
  auto size = src.size() * sizeof(T);
  if (UNLIKELY(size == 0)) return;

  if (dst_place.GetType() == phi::AllocationType::CPU) {
    VLOG(4) << "src_ptr: " << src_ptr << ", dst_ptr: " << dst_ptr << ", size: " << size;
    std::memcpy(dst_ptr, src_ptr, size);
  }
  /* TODO: 
  else if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    memory::Copy(
        dst_place, dst_ptr, src_place, src_ptr, size,
        reinterpret_cast<const phi::CustomDeviceContext&>(dev_ctx).stream());
  }
  */ 
  else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}



}  // namespace custom_kernel 

