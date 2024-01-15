cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_allocator LANGUAGES CXX CUDA)

# set your cuda architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# set NVCC compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")

# set executable
add_executable(test_allocator test/allocator_test.cu src/allocator.cu)

# link cuda library
target_link_libraries(test_allocator cuda)