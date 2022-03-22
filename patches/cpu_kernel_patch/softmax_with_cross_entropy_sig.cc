#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

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

}  // namespace phi
PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy,
                           phi::SoftmaxWithCrossEntropyOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy_grad,
                           phi::SoftmaxWithCrossEntropyGradOpArgumentMapping);

