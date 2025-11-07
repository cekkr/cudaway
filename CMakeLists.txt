cmake_minimum_required(VERSION 3.20)
project(cudaway LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(cudaway_core
    src/host/HostApiLayer.cpp
    src/device/PtxCompiler.cpp
)

target_include_directories(cudaway_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(cudaway_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(cudaway_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(cudaway_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cudaway
    src/main.cpp
)

target_link_libraries(cudaway PRIVATE cudaway_core)
