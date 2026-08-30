#include <minirt/core/cuda_error.hpp>
#include <runtime/cuda_event.hpp>
#include <runtime/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using minirt::internal::CudaEvent;
using minirt::internal::CudaStream;

#define MINIRT_TEST_REQUIRE(condition)                                                             \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::cerr << __FILE__ << ':' << __LINE__ << ": requirement failed: " #condition "\n";  \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (false)

static_assert(!std::is_copy_constructible_v<CudaStream>);
static_assert(!std::is_copy_assignable_v<CudaStream>);
static_assert(std::is_nothrow_move_constructible_v<CudaStream>);
static_assert(std::is_nothrow_move_assignable_v<CudaStream>);
static_assert(std::is_nothrow_destructible_v<CudaStream>);

static_assert(!std::is_copy_constructible_v<CudaEvent>);
static_assert(!std::is_copy_assignable_v<CudaEvent>);
static_assert(std::is_nothrow_move_constructible_v<CudaEvent>);
static_assert(std::is_nothrow_move_assignable_v<CudaEvent>);
static_assert(std::is_nothrow_destructible_v<CudaEvent>);

int main() {
    try {
        CudaStream source_stream;
        const cudaStream_t stream_handle = source_stream.get();
        MINIRT_TEST_REQUIRE(stream_handle != nullptr);

        CudaStream moved_stream(std::move(source_stream));
        MINIRT_TEST_REQUIRE(source_stream.get() == nullptr);
        MINIRT_TEST_REQUIRE(moved_stream.get() == stream_handle);

        CudaStream assigned_stream;
        assigned_stream = std::move(moved_stream);
        MINIRT_TEST_REQUIRE(moved_stream.get() == nullptr);
        MINIRT_TEST_REQUIRE(assigned_stream.get() == stream_handle);

        assigned_stream.reset();
        MINIRT_TEST_REQUIRE(assigned_stream.get() == nullptr);
        assigned_stream.reset();

        CudaEvent source_event;
        const cudaEvent_t event_handle = source_event.get();
        MINIRT_TEST_REQUIRE(event_handle != nullptr);

        CudaEvent moved_event(std::move(source_event));
        MINIRT_TEST_REQUIRE(source_event.get() == nullptr);
        MINIRT_TEST_REQUIRE(moved_event.get() == event_handle);

        CudaEvent assigned_event;
        assigned_event = std::move(moved_event);
        MINIRT_TEST_REQUIRE(moved_event.get() == nullptr);
        MINIRT_TEST_REQUIRE(assigned_event.get() == event_handle);

        assigned_event.reset();
        MINIRT_TEST_REQUIRE(assigned_event.get() == nullptr);
        assigned_event.reset();

        bool error_was_thrown = false;
        try {
            MINIRT_CUDA_CHECK(cudaErrorInvalidValue);
        } catch (const std::runtime_error& error) {
            error_was_thrown = true;
            const std::string message = error.what();

            MINIRT_TEST_REQUIRE(message.find("cudaErrorInvalidValue") != std::string::npos);
            MINIRT_TEST_REQUIRE(message.find(cudaGetErrorString(cudaErrorInvalidValue)) !=
                                std::string::npos);
        }
        MINIRT_TEST_REQUIRE(error_was_thrown);
    } catch (const std::exception& error) {
        std::cerr << "Runtime wrapper test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
