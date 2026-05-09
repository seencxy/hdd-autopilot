cmake_minimum_required(VERSION 3.24)

project(mining-opencl LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
    )
endif()

add_compile_definitions(HAVE_OPENCL=1 CL_TARGET_OPENCL_VERSION=120)

set(ARGON2_FILL_BLOCK_SOURCE third_party/argon2/src/ref.c)
if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64"
    OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
    set(ARGON2_FILL_BLOCK_SOURCE third_party/argon2/src/opt.c)
endif()

add_library(argon2_ref STATIC
    third_party/argon2/src/argon2.c
    third_party/argon2/src/core.c
    third_party/argon2/src/encoding.c
    ${ARGON2_FILL_BLOCK_SOURCE}
    third_party/argon2/src/thread.c
    third_party/argon2/src/blake2/blake2b.c
)
target_include_directories(argon2_ref
    PUBLIC third_party/argon2/include
    PRIVATE third_party/argon2/src third_party/argon2/src/blake2
)
if(MSVC)
    target_compile_options(argon2_ref PRIVATE /arch:AVX2)
endif()

add_library(argon2_gpu_common STATIC
    third_party/argon2-gpu/src/argon2-gpu-common/argon2params.cpp
    third_party/argon2-gpu/src/argon2-gpu-common/blake2b.cpp
)
target_include_directories(argon2_gpu_common
    PUBLIC third_party/argon2-gpu/include
    PRIVATE
        third_party/argon2-gpu/include/argon2-gpu-common
        third_party/argon2-gpu/src/argon2-gpu-common
)

add_library(argon2_opencl STATIC
    third_party/argon2-gpu/src/argon2-opencl/device.cpp
    third_party/argon2-gpu/src/argon2-opencl/globalcontext.cpp
    third_party/argon2-gpu/src/argon2-opencl/kernelloader.cpp
    third_party/argon2-gpu/src/argon2-opencl/programcontext.cpp
    third_party/argon2-gpu/src/argon2-opencl/processingunit.cpp
    third_party/argon2-gpu/src/argon2-opencl/kernelrunner.cpp
)
target_include_directories(argon2_opencl
    PUBLIC third_party/argon2-gpu/include include
    PRIVATE
        third_party/argon2-gpu/include/argon2-opencl
        third_party/argon2-gpu/include/argon2-gpu-common
        third_party/argon2-gpu/src/argon2-opencl
        ${OpenCL_INCLUDE_DIRS}
)
target_link_libraries(argon2_opencl PUBLIC argon2_gpu_common OpenCL::OpenCL)

add_library(mining_opencl_core STATIC
    src/job.cpp
    src/solver.cpp
    src/ffi.cpp
)
target_include_directories(mining_opencl_core
    PUBLIC include third_party/argon2/include third_party/argon2-gpu/include
    PRIVATE ${OpenCL_INCLUDE_DIRS}
)
target_link_libraries(mining_opencl_core PUBLIC argon2_ref argon2_opencl)
