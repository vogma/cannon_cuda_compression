cmake_minimum_required(VERSION 3.18)

project(Cannons_Algorithm VERSION 0.1 LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread program_options)

#cuda library for matrix multiplications
add_library(dgemm_cuda cudaMatrixMultiply.cu cudaMatrixMultiply.h)
target_include_directories(dgemm_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

#ndzip build options
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-march=native" CACHE STRING "" FORCE)

#ndzip submodule
add_subdirectory(ndzip)
target_compile_options(ndzip PUBLIC -Wno-unused-parameter)

#shared library for ndzip compression and decompression calls
add_library(ndzip_api SHARED ndzip_api.h ndzip_api.cc)
set_property(TARGET ndzip_api PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(ndzip_api PRIVATE cxx_std_17)
target_include_directories(ndzip_api PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ndzip/include")
target_include_directories(ndzip_api PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ndzip/src")
target_link_libraries(ndzip_api ndzip ndzip-cuda)

#implementation of cannons algorithm
add_executable(Cannons_Algorithm cannon_no_comp.cpp matrix_functions.h cudaMatrixMultiply.h)
target_compile_features(Cannons_Algorithm PRIVATE cxx_std_17)
set_property(TARGET Cannons_Algorithm PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(Cannons_Algorithm_Comp cannon_comp.cpp matrix_functions.h cudaMatrixMultiply.h)
target_compile_features(Cannons_Algorithm_Comp PRIVATE cxx_std_17)
set_property(TARGET Cannons_Algorithm_Comp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(Cannons_Algorithm_Comp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ndzip/src" "${CMAKE_CURRENT_SOURCE_DIR}/ndzip/include")
target_link_libraries(Cannons_Algorithm_Comp PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver ndzip ndzip-cuda ndzip_api dgemm_cuda)


target_include_directories(Cannons_Algorithm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ndzip/src" "${CMAKE_CURRENT_SOURCE_DIR}/ndzip/include")
target_link_libraries(Cannons_Algorithm PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver ndzip ndzip-cuda ndzip_api dgemm_cuda)
