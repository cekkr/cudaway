cmake_minimum_required(VERSION 3.20)
project(cudaway LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HipWindowsWorkarounds)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(cudaway_core
    src/host/CudaDriverShim.cpp
    src/host/HostApiLayer.cpp
    src/host/runtime/RuntimeRegistry.cpp
    src/device/PtxCompiler.cpp
    src/platform/PlatformConfig.cpp
)

target_include_directories(cudaway_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(cudaway_core PUBLIC cxx_std_20)

target_compile_definitions(cudaway_core
    PRIVATE
        $<$<PLATFORM_ID:Windows>:CUDAWAY_PLATFORM_WINDOWS=1>
        $<$<NOT:$<PLATFORM_ID:Windows>>:CUDAWAY_PLATFORM_LINUX=1>
)

if(MSVC)
    target_compile_options(cudaway_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(cudaway_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32 AND TARGET cudaway_hip_windows)
    target_link_libraries(cudaway_core PRIVATE cudaway_hip_windows)
endif()

add_executable(cudaway
    src/main.cpp
)

target_link_libraries(cudaway PRIVATE cudaway_core)
