#pragma once

#include <cuda_runtime.h>

#include <string_view>

namespace minirt::details {
void CheckCudaNoThrow(cudaError_t result, std::string_view operation, std::string_view file,
                      int line) noexcept;
void CheckCuda(cudaError_t result, std::string_view operation, std::string_view file, int line);
} // namespace minirt::details

#define MINIRT_CUDA_CHECK(expression)                                                              \
    do {                                                                                           \
        ::minirt::details::CheckCuda((expression), #expression, __FILE__, __LINE__);               \
    } while (false)

#define MINIRT_CUDA_CHECK_NO_THROW(expression)                                                     \
    do {                                                                                           \
        ::minirt::details::CheckCudaNoThrow((expression), #expression, __FILE__, __LINE__);        \
    } while (false)
