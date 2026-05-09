cmake_minimum_required(VERSION 3.24)

project(mining-metal LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "Metal backend requires macOS")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DIST_DIR "${CMAKE_SOURCE_DIR}/../../artifacts")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${DIST_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${DIST_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${CONFIG}" CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${DIST_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${DIST_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/lib")
endforeach()

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

set(ARGON2_FILL_BLOCK_SOURCE ../mining-opencl/third_party/argon2/src/ref.c)
if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64"
    OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
    set(ARGON2_FILL_BLOCK_SOURCE ../mining-opencl/third_party/argon2/src/opt.c)
endif()

add_library(argon2_ref STATIC
    ../mining-opencl/third_party/argon2/src/argon2.c
    ../mining-opencl/third_party/argon2/src/core.c
    ../mining-opencl/third_party/argon2/src/encoding.c
    ${ARGON2_FILL_BLOCK_SOURCE}
    ../mining-opencl/third_party/argon2/src/thread.c
    ../mining-opencl/third_party/argon2/src/blake2/blake2b.c
)
target_include_directories(argon2_ref
    PUBLIC ../mining-opencl/third_party/argon2/include
    PRIVATE
        ../mining-opencl/third_party/argon2/src
        ../mining-opencl/third_party/argon2/src/blake2
)

add_library(argon2_gpu_common STATIC
    ../mining-opencl/third_party/argon2-gpu/src/argon2-gpu-common/argon2params.cpp
    ../mining-opencl/third_party/argon2-gpu/src/argon2-gpu-common/blake2b.cpp
)
target_include_directories(argon2_gpu_common
    PUBLIC ../mining-opencl/third_party/argon2-gpu/include
    PRIVATE
        ../mining-opencl/third_party/argon2-gpu/include/argon2-gpu-common
        ../mining-opencl/third_party/argon2-gpu/src/argon2-gpu-common
)

add_library(mining_metal_core STATIC
    src/job.cpp
    src/solver.mm
    src/ffi.mm
)
target_compile_options(mining_metal_core PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
)
target_include_directories(mining_metal_core
    PUBLIC
        include
        ../mining-opencl/third_party/argon2/include
        ../mining-opencl/third_party/argon2-gpu/include
)
target_link_libraries(mining_metal_core
    PUBLIC
        argon2_ref
        argon2_gpu_common
        ${FOUNDATION_FRAMEWORK}
        ${METAL_FRAMEWORK}
)
