cmake_minimum_required(VERSION 3.27)

project(mnistdata LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mnistdata mnistdata.cpp)
