#pragma once

#include <minirt/core/cuda_error.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace minirt::memory {

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(std::size_t count) {
        MINIRT_CUDA_CHECK(cudaMalloc(&bufferDevice_, count * sizeof(T)));
        size_ = count;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) {
        bufferDevice_ = std::exchange(other.bufferDevice_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) {
        if (std::addressof(other) == this) {
            return *this;
        }

        reset();
        bufferDevice_ = std::exchange(other.bufferDevice_, nullptr);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    [[nodiscard]] T* get() const { return bufferDevice_; }

    [[nodiscard]] std::size_t size() const { return size_; }

    void reset() {
        if (bufferDevice_) {
            MINIRT_CUDA_CHECK_NO_THROW(cudaFree(bufferDevice_));
            size_ = 0;
            bufferDevice_ = nullptr;
        }
    }

    ~DeviceBuffer() { reset(); }

  private:
    T* bufferDevice_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace minirt::memory
