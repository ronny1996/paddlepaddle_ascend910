#include <iostream>
#include "paddle/extension.h"
#include "paddle/pten/api/ext/op_kernel_api.h"

static void ReluOp_Compute(const paddle::PD_ExecutionContext* ctx) {
  int num = PD_NumInputs(ctx);
  std::cout << "input num = " << num << std::endl;
}

PD_BUILD_KERNEL(relu)
    .SetKernelFn(ReluOp_Compute);
