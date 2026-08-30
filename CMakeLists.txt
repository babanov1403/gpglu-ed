cmake_minimum_required(VERSION 3.25)

set(
    CMAKE_CUDA_ARCHITECTURES
    "75"
    CACHE STRING
    "CUDA architectures to build for"
)

project(
    gpglu_ed
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(
    minirt_runtime
    STATIC
    src/internal/cuda_error.cpp
    src/internal/cuda_event.cpp
    src/internal/cuda_stream.cpp
)

add_library(minirt::runtime ALIAS minirt_runtime)

target_include_directories(
    minirt_runtime
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src/internal"
)

target_link_libraries(
    minirt_runtime
    PUBLIC
        CUDA::cudart
)

include(CTest)

if(BUILD_TESTING)
    add_executable(
        minirt_cuda_smoke
        tests/cuda_smoke.cu
    )
    target_link_libraries(
        minirt_cuda_smoke
        PRIVATE
            minirt::runtime
    )
    add_test(NAME cuda_smoke COMMAND minirt_cuda_smoke)

    add_executable(
        minirt_runtime_wrappers_test
        tests/runtime_wrappers_test.cpp
    )
    target_link_libraries(
        minirt_runtime_wrappers_test
        PRIVATE
            minirt::runtime
    )
    add_test(
        NAME runtime_wrappers
        COMMAND minirt_runtime_wrappers_test
    )

    add_executable(
        minirt_public_headers_compile
        tests/public_headers_compile.cpp
    )
    target_link_libraries(
        minirt_public_headers_compile
        PRIVATE
            minirt::runtime
    )
    add_test(
        NAME public_headers_compile
        COMMAND minirt_public_headers_compile
    )

    function(minirt_verify_internal_headers_are_private probe_source)
        get_target_property(
            public_include_directories
            minirt_runtime
            INTERFACE_INCLUDE_DIRECTORIES
        )

        foreach(include_directory IN LISTS public_include_directories)
            if(include_directory MATCHES "^${PROJECT_SOURCE_DIR}/src($|/)")
                message(
                    FATAL_ERROR
                    "minirt_runtime exposes an internal include directory: "
                    "${include_directory}"
                )
            endif()
        endforeach()

        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
        try_compile(
            internal_header_is_visible
            SOURCES "${probe_source}"
            CMAKE_FLAGS
                "-DINCLUDE_DIRECTORIES=${public_include_directories};${CUDAToolkit_INCLUDE_DIRS}"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            NO_CACHE
            OUTPUT_VARIABLE probe_output
        )

        if(internal_header_is_visible)
            message(
                FATAL_ERROR
                "The backend-facing include interface can compile an internal "
                "header. Probe output:\n${probe_output}"
            )
        endif()
    endfunction()

    minirt_verify_internal_headers_are_private(
        "${PROJECT_SOURCE_DIR}/tests/negative_internal_include.cpp"
    )
endif()
