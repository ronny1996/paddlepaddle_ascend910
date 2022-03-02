#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature ReduceMaxOpArgumentMapping(const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature("max", {"X"}, {"dim", "out_dtype", "keep_dim"},
                             {"Out"});
    }
    return KernelSignature("max_raw", {"X"},
                           {"dim", "keep_dim", "reduce_all", "out_dtype"},
                           {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ReduceMinOpArgumentMapping(const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature("min", {"X"}, {"dim", "out_dtype", "keep_dim"},
                             {"Out"});
    }
    return KernelSignature("min_raw", {"X"},
                           {"dim", "keep_dim", "reduce_all", "out_dtype"},
                           {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ReduceMeanGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
  if (ctx.IsDenseTensorInput("X")) {
    if (!reduce_all) {
      return KernelSignature("mean_grad", {"X", GradVarName("Out")},
                             {"dim", "keep_dim"}, {GradVarName("X")});
    }
    return KernelSignature("mean_raw_grad", {"X", GradVarName("Out")},
                           {"dim", "keep_dim", "reduce_all"},
                           {GradVarName("X")});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature SliceOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("slice", {"Input"}, {"axes", "starts", "ends"},
                         {"Out"});
}

KernelSignature SGDOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("sgd", {"Param", "LearningRate", "Grad"}, {},
                         {"ParamOut"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(reduce_max, max);
PD_REGISTER_BASE_KERNEL_NAME(reduce_min, min);
PD_REGISTER_BASE_KERNEL_NAME(reduce_mean_grad, mean_grad);

PD_REGISTER_ARG_MAPPING_FN(reduce_max, phi::ReduceMaxOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reduce_min, phi::ReduceMinOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(reduce_mean_grad,
                           phi::ReduceMeanGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(slice, phi::SliceOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sgd, phi::SGDOpArgumentMapping);
