#include <minirt/core/cuda_error.hpp>
#include <minirt/internal/memory/device_buffer.hpp>
#include <minirt/test/test_require.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace minirt::memory;

__global__ void MultiplyKernel(int* bufferDevice_, int size) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size) {
        bufferDevice_[idx] *= 2;
    }
}

void TestTypeTraits() {
    static_assert(!std::is_copy_assignable_v<DeviceBuffer<float>>);
    static_assert(!std::is_copy_constructible_v<DeviceBuffer<float>>);

    static_assert(std::is_move_assignable_v<DeviceBuffer<float>>);
    static_assert(std::is_move_constructible_v<DeviceBuffer<float>>);
}

void TestEmptyBuffer() {
    DeviceBuffer<float> buf;
    MINIRT_TEST_REQUIRE(buf.get() == nullptr);
}

void TestAllocationAndCopy() {
    std::array<int, 5> arrayHost = {1, 2, 3, 4, 5};
    DeviceBuffer<int> bufDevice(5);

    MINIRT_CUDA_CHECK(
        cudaMemcpy(bufDevice.get(), arrayHost.data(), 5 * sizeof(int), cudaMemcpyHostToDevice));

    MultiplyKernel<<<1, 5>>>(bufDevice.get(), bufDevice.size());
    MINIRT_CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    MINIRT_CUDA_CHECK(
        cudaMemcpy(arrayHost.data(), bufDevice.get(), 5 * sizeof(int), cudaMemcpyDeviceToHost));
    const std::array<int, 5> expected = {2, 4, 6, 8, 10};
    MINIRT_TEST_REQUIRE(arrayHost == expected);
}

void TestMoveConstruction() {
    DeviceBuffer<float> buf1(5);
    DeviceBuffer<float> buf2 = std::move(buf1);
}

void TestMoveAssignment() {
    DeviceBuffer<float> buf1(5);
    DeviceBuffer<float> buf2;

    buf2 = std::move(buf1);
    MINIRT_TEST_REQUIRE(buf1.get() == nullptr);
    MINIRT_TEST_REQUIRE(buf1.size() == 0);

    MINIRT_TEST_REQUIRE(buf2.size() == 5);
    MINIRT_TEST_REQUIRE(buf2.get() != nullptr);
}

void TestReset() {
    DeviceBuffer<double> buf(10);

    MINIRT_TEST_REQUIRE(buf.size() == 10);
    buf.reset();

    MINIRT_TEST_REQUIRE(buf.size() == 0);
    MINIRT_TEST_REQUIRE(buf.get() == nullptr);
}

int main() {
    try {
        TestTypeTraits();
        TestEmptyBuffer();
        TestAllocationAndCopy();
        TestMoveConstruction();
        TestMoveAssignment();
        TestReset();

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "DeviceBuffer test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
