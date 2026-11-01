cmake_minimum_required(VERSION 3.25)
project(gpusr LANGUAGES CXX)

file(GLOB_RECURSE sources src/*.cpp)
file(GLOB_RECURSE headers src/*.hpp)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sample ${headers} ${sources})

