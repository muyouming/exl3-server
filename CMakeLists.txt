cmake_minimum_required(VERSION 3.24)
project(exl3-server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(mlx REQUIRED)

add_executable(exl3-server
  src/main.cpp
  src/fp16.cpp
  src/safetensors.cpp
  src/exl3.cpp
  src/model.cpp
  src/tokenizer.cpp
  src/server.cpp
  src/cli.cpp
)

target_include_directories(exl3-server PRIVATE vendor src)
target_compile_options(exl3-server PRIVATE -O3 -march=native -mfp16-format=ieee)
target_link_libraries(exl3-server PRIVATE mlx::mlx Threads::Threads)

set_target_properties(exl3-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
