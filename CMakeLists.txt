cmake_minimum_required(VERSION 3.24)
project(exl3-server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# MLX (the Homebrew bottle ships no CMake package config, so locate it directly).
find_path(MLX_INCLUDE_DIR mlx/mlx.h
  HINTS ${MLX_ROOT}/include /opt/homebrew/opt/mlx/include /usr/local/opt/mlx/include)
find_library(MLX_LIBRARY NAMES mlx
  HINTS ${MLX_ROOT}/lib /opt/homebrew/opt/mlx/lib /usr/local/opt/mlx/lib)
if(NOT MLX_INCLUDE_DIR OR NOT MLX_LIBRARY)
  execute_process(COMMAND brew --prefix mlx
    OUTPUT_VARIABLE _brew_mlx OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(EXISTS "${_brew_mlx}/include/mlx/mlx.h")
    set(MLX_INCLUDE_DIR "${_brew_mlx}/include")
    find_library(MLX_LIBRARY NAMES mlx PATHS "${_brew_mlx}/lib" NO_DEFAULT_PATH)
  endif()
endif()
if(NOT MLX_INCLUDE_DIR OR NOT MLX_LIBRARY)
  message(FATAL_ERROR "MLX not found. Install it with `brew install mlx` or pass -DMLX_ROOT=<prefix>.")
endif()
get_filename_component(MLX_LIBRARY_DIR ${MLX_LIBRARY} DIRECTORY)
message(STATUS "MLX include: ${MLX_INCLUDE_DIR}")
message(STATUS "MLX library: ${MLX_LIBRARY}")

add_executable(exl3-server
  src/main.cpp
  src/safetensors.cpp
  src/exl3.cpp
  src/trellis_gemm.cpp
  src/model.cpp
  src/tokenizer.cpp
  src/engine.cpp
  src/server.cpp
  src/cli.cpp
)

target_include_directories(exl3-server PRIVATE ${MLX_INCLUDE_DIR} vendor src)
target_compile_options(exl3-server PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(exl3-server PRIVATE ${MLX_LIBRARY} Threads::Threads)
target_link_options(exl3-server PRIVATE -Wl,-rpath,${MLX_LIBRARY_DIR})

set_target_properties(exl3-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
