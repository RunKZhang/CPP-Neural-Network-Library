cmake_minimum_required(VERSION 2.8)

project(cpp_neural_network_library)
set(CMAKE_CXX_STANDARD 17)

file(GLOB srclist "src/*.h")
INCLUDE_DIRECTORIES(eigen)
INCLUDE_DIRECTORIES(autodiff)

add_executable(mymathexe src/exemain.cpp ${srclist})
