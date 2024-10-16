cmake_minimum_required(VERSION 3.27)

project(fast_autodiff LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(fast_autodiff fast_autodiff.cu)
target_link_libraries(fast_autodiff PUBLIC cublas )
