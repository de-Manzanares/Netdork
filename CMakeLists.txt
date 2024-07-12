cmake_minimum_required(VERSION 3.20)

project(Netdork)

set(CMAKE_CXX_STANDARD 23)

enable_testing()
include(CTest)

add_subdirectory(mnist)
add_subdirectory(xor)
