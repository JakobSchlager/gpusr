cmake_minimum_required(VERSION 3.25)
project(gpusr LANGUAGES CXX)

# TODO: clean this up
# file(GLOB_RECURSE sources src/*.cpp)
# file(GLOB_RECURSE headers src/*.hpp)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GPUSR_ENABLE_CUDA "Enable CUDA evaluator" ON)

# add_executable(gpusr ${headers} ${sources})
add_executable(gpusr 
    src/dataparser.hpp
    src/dataset.h
    src/evaluator.h
    src/evolution.hpp
    src/exporter.hpp
    src/operators.hpp
    src/parameters.h
    src/main.cpp
)

add_executable(ad
    src/ad.cpp
    src/operators.hpp
)

if(GPUSR_ENABLE_CUDA)
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    target_sources(gpusr PRIVATE
        src/evaluator_cuda.cu
    )

    set_target_properties(gpusr PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 75
    )

    target_link_libraries(gpusr PRIVATE
        CUDA::cudart
    )

    target_compile_definitions(gpusr PRIVATE
        GPUSR_ENABLE_CUDA=1
    )
else()
    target_sources(gpusr PRIVATE 
	src/evaluator.cpp)
endif()
