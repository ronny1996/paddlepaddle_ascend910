#pragma once

#include <acl/acl.h>

#include "paddle/phi/extension.h"

#define ACL_CHECK(func)                                                       \
  do {                                                                        \
    auto acl_ret = func;                                                      \
    if (acl_ret != ACL_ERROR_NONE) {                                          \
      std::cerr << "Call " << #func << " failed : " << acl_ret << " at file " \
                << __FILE__ << " line " << __LINE__ << std::endl;             \
      {                                                                       \
        const char *aclRecentErrMsg = nullptr;                                \
        aclRecentErrMsg = aclGetRecentErrMsg();                               \
        if (aclRecentErrMsg != nullptr) {                                     \
          printf("%s\n", aclRecentErrMsg);                                    \
        } else {                                                              \
          printf("Failed to get recent error message.\n");                    \
        }                                                                     \
      }                                                                       \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)


C_Status MemCpyH2D(const C_Device device, void *dst, const void *src,
                   size_t size);
C_Status MemCpyD2D(const C_Device device, void *dst, const void *src,
                   size_t size);
C_Status MemCpyD2H(const C_Device device, void *dst, const void *src,
                   size_t size);
C_Status AsyncMemCpyH2D(const C_Device device, C_Stream stream, void *dst,
                        const void *src, size_t size);
C_Status AsyncMemCpyD2D(const C_Device device, C_Stream stream, void *dst,
                        const void *src, size_t size);
C_Status AsyncMemCpyD2H(const C_Device device, C_Stream stream, void *dst,
                        const void *src, size_t size);
