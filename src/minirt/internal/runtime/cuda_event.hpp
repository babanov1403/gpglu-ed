#pragma once

#include <minirt/core/cuda_error.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <utility>

namespace minirt::internal {
class CudaEvent {
  public:
    CudaEvent() { MINIRT_CUDA_CHECK(cudaEventCreate(&event_)); }

    CudaEvent(CudaEvent const&) = delete;
    CudaEvent& operator=(CudaEvent const&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (std::addressof(other) == this) {
            return *this;
        }

        reset();
        event_ = std::exchange(other.event_, nullptr);
        return *this;
    }

    void reset() noexcept {
        if (event_ == nullptr) {
            return;
        }

        cudaEvent_t event = std::exchange(event_, nullptr);
        MINIRT_CUDA_CHECK_NO_THROW(cudaEventDestroy(event));
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    ~CudaEvent() { reset(); }

  private:
    cudaEvent_t event_{nullptr};
};
} // namespace minirt::internal
