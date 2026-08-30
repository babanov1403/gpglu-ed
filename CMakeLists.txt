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
    src/internal/runtime/cuda_event.cpp
    src/internal/runtime/cuda_stream.cpp
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

find_program(MINIRT_CLANG_FORMAT_EXECUTABLE NAMES clang-format)

if(MINIRT_CLANG_FORMAT_EXECUTABLE)
    file(
        GLOB_RECURSE MINIRT_FORMAT_SOURCES
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hh"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/include/*.cuh"
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cc"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cxx"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/src/*.cuh"
        "${PROJECT_SOURCE_DIR}/src/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.hh"
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        "${PROJECT_SOURCE_DIR}/tests/*.cc"
        "${PROJECT_SOURCE_DIR}/tests/*.cpp"
        "${PROJECT_SOURCE_DIR}/tests/*.cxx"
        "${PROJECT_SOURCE_DIR}/tests/*.cu"
        "${PROJECT_SOURCE_DIR}/tests/*.cuh"
        "${PROJECT_SOURCE_DIR}/tests/*.h"
        "${PROJECT_SOURCE_DIR}/tests/*.hh"
        "${PROJECT_SOURCE_DIR}/tests/*.hpp"
    )

    add_custom_target(
        format
        COMMAND
            "${MINIRT_CLANG_FORMAT_EXECUTABLE}"
            --style=file
            -i
            ${MINIRT_FORMAT_SOURCES}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Formatting C++ and CUDA sources"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )

    add_custom_target(
        format-check
        COMMAND
            "${MINIRT_CLANG_FORMAT_EXECUTABLE}"
            --style=file
            --dry-run
            --Werror
            ${MINIRT_FORMAT_SOURCES}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Checking C++ and CUDA source formatting"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
else()
    message(STATUS "clang-format not found; format targets are unavailable")
endif()

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
