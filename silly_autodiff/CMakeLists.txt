cmake_minimum_required(VERSION 3.27)

project(silly_autodiff LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(silly_autodiff silly_autodiff.cu)
target_link_libraries(silly_autodiff PUBLIC cublas )
