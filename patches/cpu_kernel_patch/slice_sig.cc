#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature SliceOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("slice", {"Input"}, {"axes", "starts", "ends"},
                         {"Out"});
}

KernelSignature SliceGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("slice_grad", {"Input", GradVarName("Out")},
                         {"axes", "starts", "ends"}, {GradVarName("Input")});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(slice, phi::SliceOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(slice_grad, phi::SliceGradOpArgumentMapping);

