cmake_minimum_required(VERSION 3.10)
project(gpusr)  

file(GLOB_RECURSE sources src/*.cpp)
file(GLOB_RECURSE headers src/*.h)

set(CMAKE_CXX_STANDARD 26)

add_executable(sample ${headers} ${sources})

if (MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    # std:: print, println, ...
    target_link_libraries(sample stdc++exp) 
endif()