#include <minirt/core/cuda_error.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace minirt::details {

void CheckCudaNoThrow(cudaError_t result, std::string_view operation, std::string_view file,
                      int line) noexcept {
    if (result == cudaSuccess) {
        return;
    }

    (void)std::fprintf(stderr, "%s:%d: CUDA operation `%s` failed with %s: %s\n", file.data(), line,
                       operation.data(), cudaGetErrorName(result), cudaGetErrorString(result));
}

void CheckCuda(cudaError_t result, std::string_view operation, std::string_view file, int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::string message;
    message += file;
    message += ':';
    message += std::to_string(line);
    message += ": CUDA operation `";
    message += operation;
    message += "` failed with ";
    message += cudaGetErrorName(result);
    message += ": ";
    message += cudaGetErrorString(result);

    throw std::runtime_error(message);
}
} // namespace minirt::details