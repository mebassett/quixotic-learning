cmake_minimum_required(VERSION 3.27)

project(silly-autodiff LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(fast_autodiff)
add_subdirectory(mnistdata)

add_executable(ffnn ffnn/ffnn-fast_autodiff.cu)
target_include_directories(ffnn PUBLIC ${PROJECT_SOURCE_DIR}/fast_autodiff)
target_include_directories(ffnn PUBLIC ${PROJECT_SOURCE_DIR}/mnist)
target_link_libraries(ffnn PUBLIC cublas fast_autodiff mnistdata)

add_executable(lenet lenet/lenet.cu)
target_include_directories(lenet PUBLIC ${PROJECT_SOURCE_DIR}/fast_autodiff)
target_include_directories(lenet PUBLIC ${PROJECT_SOURCE_DIR}/mnist)
target_link_libraries(lenet PUBLIC cublas fast_autodiff mnistdata)
