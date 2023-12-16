# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 2.8.12)
project(llama2.c-to-ncnn)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 17)

# 设置程序编译出的可执行文件
set(MAIN_FILE inference.cpp convert.cpp)
set(EXECUTABLE_OUTPUT_PATH ../)

if(DEFINED ENV{NCNN_DIR})
    message("NCNN_DIR = $ENV{NCNN_DIR}")
    include($ENV{NCNN_DIR}/build/install/lib/cmake/ncnn/ncnnConfig.cmake)
else()
    message("NCNN_DIR not set,find ncnn in parent directory")
    include(../ncnn/build/install/lib/cmake/ncnn/ncnnConfig.cmake)
endif()

# 配置OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_executable(inference inference.cpp)
add_executable(convert convert.cpp)

if(NCNN_VULKAN)
    target_link_libraries(inference glslang::glslang ncnn)
else()
    target_link_libraries(inference ncnn)
endif()
