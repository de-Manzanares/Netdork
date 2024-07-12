project(Netdork)

set(CMAKE_CXX_STANDARD 23)

find_package(Catch2 3 QUIET)

file(GLOB ANN_FRAMEWORK "../ann_framework/*.cpp")

add_executable(mnist-training
        mnist-training.cpp
        ${ANN_FRAMEWORK})
target_include_directories(mnist-training PRIVATE ../ann_framework)

add_executable(mnist-testing
        mnist-testing.cpp
        ${ANN_FRAMEWORK})
target_include_directories(mnist-testing PRIVATE ../ann_framework)
