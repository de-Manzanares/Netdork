project(Netdork)

set(CMAKE_CXX_STANDARD 23)

find_package(Catch2 3 QUIET)

add_executable(xorNet-test
        xor-test.cxx
        ../src/Neuron.cpp
        ../src/Net.cpp)
target_include_directories(xorNet-test PRIVATE ../include)
target_link_libraries(xorNet-test PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)

add_test(NAME xorNetTest COMMAND xorNet-test)
