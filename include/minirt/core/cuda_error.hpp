#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <string_view>

namespace minirt::details {
    std::string CheckCudaNoThrow(cudaError_t result, std::string_view operation, std::string_view file, int line) noexcept;
    std::string CheckCuda(cudaError_t result, std::string_view operation, std::string_view file, int line);
}

#define CUDA_CHECK(expression)                        \
    do {                                              \
        ::minirt::detail::CheckCuda(        \
            (expression),                             \
            #expression,                              \
            __FILE__,                                 \
            __LINE__);                                \
    } while (false)

#define CUDA_CHECK_NO_THROW(expression)                        \
    do {                                              \
        ::minirt::detail::CheckCudaNoThrow(        \
            (expression),                             \
            #expression,                              \
            __FILE__,                                 \
            __LINE__);                                \
    } while (false)
