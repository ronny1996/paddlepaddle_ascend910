#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

// KernelSignature ReduceMinOpArgumentMapping(const ArgumentMappingContext& ctx) {
//   bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
//   if (ctx.IsDenseTensorInput("X")) {
//     if (!reduce_all) {
//       return KernelSignature("min", {"X"}, {"dim", "out_dtype", "keep_dim"},
//                              {"Out"});
//     }
//     return KernelSignature("min_raw", {"X"},
//                            {"dim", "keep_dim", "reduce_all", "out_dtype"},
//                            {"Out"});
//   }
//   return KernelSignature("unregistered", {}, {}, {});
// }

// KernelSignature ReduceMeanGradOpArgumentMapping(
//     const ArgumentMappingContext& ctx) {
//   bool reduce_all = paddle::any_cast<bool>(ctx.Attr("reduce_all"));
//   if (ctx.IsDenseTensorInput("X")) {
//     if (!reduce_all) {
//       return KernelSignature("mean_grad", {"X", GradVarName("Out")},
//                              {"dim", "keep_dim"}, {GradVarName("X")});
//     }
//     return KernelSignature("mean_raw_grad", {"X", GradVarName("Out")},
//                            {"dim", "keep_dim", "reduce_all"},
//                            {GradVarName("X")});
//   }
//   return KernelSignature("unregistered", {}, {}, {});
// }

KernelSignature SliceOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("slice", {"Input"}, {"axes", "starts", "ends"},
                         {"Out"});
}

KernelSignature SliceGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("slice_grad", {"Input", GradVarName("Out")},
                         {"axes", "starts", "ends"}, {GradVarName("Input")});
}

KernelSignature AdamOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "adam", {"Param", "Grad", "LearningRate", "Moment1", "Moment2",
               "Beta1Pow", "Beta2Pow"},
      {"beta1", "beta2", "epsilon", "lazy_mode",
       "min_row_size_to_use_multithread", "multi_precision",
       "use_global_beta_pow"},
      {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut"});
}

KernelSignature MomentOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "momentum", {"Param", "Grad", "Velocity", "LearningRate"},
      {"mu", "use_nesterov", "regularization_method", "regularization_coeff",
       "multi_precision", "rescale_grad"},
      {"ParamOut", "VelocityOut"});
}

// KernelSignature LessThanArgumentMapping(const ArgumentMappingContext& ctx) {
//   return KernelSignature("less_than", {"X", "Y"}, {"axis"}, {"Out"});
// }

// KernelSignature GreaterEqualArgumentMapping(const ArgumentMappingContext& ctx) {
//   return KernelSignature("greater_equal", {"X", "Y"}, {"axis"}, {"Out"});
// }

// KernelSignature EqualArgumentMapping(const ArgumentMappingContext& ctx) {
//   return KernelSignature("equal", {"X", "Y"}, {"axis"}, {"Out"});
// }

// KernelSignature NotEqualArgumentMapping(const ArgumentMappingContext& ctx) {
//   return KernelSignature("not_equal", {"X", "Y"}, {"axis"}, {"Out"});
// }

KernelSignature SoftmaxWithCrossEntropyOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("softmax_with_cross_entropy", {"Logits", "Label"},
                         {"soft_label", "use_softmax", "numeric_stable_mode",
                          "ignore_index", "axis"},
                         {"Softmax", "Loss", "Backprop"});
}

KernelSignature SoftmaxWithCrossEntropyGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("softmax_with_cross_entropy_grad",
                         {"Label", "Softmax", "Backprop", GradVarName("Loss")},
                         {"soft_label", "use_softmax", "numeric_stable_mode",
                          "ignore_index", "axis"},
                         {GradVarName("Logits")});
}

KernelSignature MergedMomentOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "momentum", {"Param", "Grad", "Velocity", "LearningRate"},
      {"mu", "use_nesterov", "regularization_method", "regularization_coeff",
       "multi_precision", "rescale_grad"},
      {"ParamOut", "VelocityOut"});
}
}  // namespace phi

// PD_REGISTER_BASE_KERNEL_NAME(reduce_min, min);
// PD_REGISTER_BASE_KERNEL_NAME(reduce_mean_grad, mean_grad);
PD_REGISTER_ARG_MAPPING_FN(merged_momentum, phi::MergedMomentOpArgumentMapping);

// PD_REGISTER_ARG_MAPPING_FN(reduce_min, phi::ReduceMinOpArgumentMapping);
// PD_REGISTER_ARG_MAPPING_FN(reduce_mean_grad,
                          //  phi::ReduceMeanGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(slice, phi::SliceOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(slice_grad, phi::SliceGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(adam, phi::AdamOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(momentum, phi::MomentOpArgumentMapping);

// PD_REGISTER_ARG_MAPPING_FN(less_than, phi::LessThanArgumentMapping);
// PD_REGISTER_ARG_MAPPING_FN(greater_equal, phi::GreaterEqualArgumentMapping);
// PD_REGISTER_ARG_MAPPING_FN(equal, phi::EqualArgumentMapping);
// PD_REGISTER_ARG_MAPPING_FN(not_equal, phi::NotEqualArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy,
                           phi::SoftmaxWithCrossEntropyOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy_grad,
                           phi::SoftmaxWithCrossEntropyGradOpArgumentMapping);
