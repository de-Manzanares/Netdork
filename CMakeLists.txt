cmake_minimum_required(VERSION 3.20)

project(Netdork)

set(CMAKE_CXX_STANDARD 23)

add_executable(train-xorNet
        src/train-xorNet.cpp
        src/Neuron.cpp
        src/Net.cpp)
target_include_directories(train-xorNet PRIVATE include)

enable_testing()
include(CTest)

add_subdirectory(test)
