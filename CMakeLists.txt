cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_allocator LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

# 设置 NVCC 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")

# 指定生成的目标
add_executable(test_allocator test/allocator_test.cu src/allocator.cu)

# 链接 CUDA 库
target_link_libraries(test_allocator cuda)