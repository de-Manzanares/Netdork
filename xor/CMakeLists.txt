project(Netdork)

set(CMAKE_CXX_STANDARD 23)

find_package(Catch2 3 QUIET)

file(GLOB ANN_FRAMEWORK "../ann_framework/*.cpp")

add_executable(xor-training
        xor-training.cpp
        ${ANN_FRAMEWORK})
target_include_directories(xor-training PRIVATE ../ann_framework)

add_executable(xor-testing
        xor-testing.cxx
        ${ANN_FRAMEWORK})
target_include_directories(xor-testing PRIVATE ../ann_framework)
target_link_libraries(xor-testing PRIVATE Catch2::Catch2WithMain)


include(CTest)
include(Catch)

add_test(NAME xorTest COMMAND xor-testing)
