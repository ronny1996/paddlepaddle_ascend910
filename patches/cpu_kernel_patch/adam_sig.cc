#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature AdamOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "adam", {"Param", "Grad", "LearningRate", "Moment1", "Moment2",
               "Beta1Pow", "Beta2Pow"},
      {"beta1", "beta2", "epsilon", "lazy_mode",
       "min_row_size_to_use_multithread", "multi_precision",
       "use_global_beta_pow"},
      {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(adam, phi::AdamOpArgumentMapping);

