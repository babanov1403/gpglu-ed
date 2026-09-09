#pragma once

#include <minirt/core/cuda_error.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <utility>

namespace minirt::internal {
class CudaStream {
  public:
    CudaStream() { MINIRT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (std::addressof(other) == this) {
            return *this;
        }

        reset();
        stream_ = std::exchange(other.stream_, nullptr);
        return *this;
    }

    void reset() noexcept {
        if (stream_ == nullptr) {
            return;
        }

        cudaStream_t stream = std::exchange(stream_, nullptr);
        MINIRT_CUDA_CHECK_NO_THROW(cudaStreamDestroy(stream));
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    ~CudaStream() { reset(); }

  private:
    cudaStream_t stream_{nullptr};
};
} // namespace minirt::internal
