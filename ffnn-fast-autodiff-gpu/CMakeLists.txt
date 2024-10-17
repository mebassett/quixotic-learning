cmake_minimum_required(VERSION 3.27)

project(silly-autodiff LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CLANG_BUILD_NEW_DRIVER on)
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

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    tests
    tests/fast_autodiff.test.cu
)
target_link_libraries(
    tests
    PRIVATE
    GTest::gtest_main
    fast_autodiff
)
target_include_directories(
    tests
    PRIVATE
    "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include"
    ${PROJECT_SOURCE_DIR}/fast_autodiff
)
set_target_properties(tests PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_ARCHITECTURES "52")
set_target_properties(tests PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_ARCHITECTURES "52"
       CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}
   )

target_compile_options(tests PRIVATE
       $<$<COMPILE_LANGUAGE:CUDA>:-I${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include>
   )
include(GoogleTest)

gtest_discover_tests(tests)

