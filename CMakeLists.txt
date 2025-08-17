cmake_minimum_required(VERSION 3.20)
project(MicroTransformerOpenMP VERSION 1.0.0 LANGUAGES CXX)

# Set C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compiler
if(WIN32)
    set(CMAKE_CXX_COMPILER "C:/Users/$ENV{USERNAME}/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/g++.exe")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Compilation options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Explicitly specify source files to ensure proper compilation order
set(SOURCES
    src/matrix.cpp
    src/attention.cpp  
    src/layers.cpp
    src/encoder.cpp
    src/benchmark.cpp
    src/main.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link OpenMP
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Compile options
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create build directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Test targets
enable_testing()

# Basic functionality test
add_custom_target(run_tests
    COMMAND $<TARGET_FILE:${PROJECT_NAME}>
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running transformer tests"
)

# Performance benchmark with different thread counts
add_custom_target(benchmark
    COMMAND echo "=== Running Performance Benchmarks ===" &&
            echo "1 Thread:" &&
            ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1 $<TARGET_FILE:${PROJECT_NAME}> &&
            echo "" && echo "2 Threads:" &&
            ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=2 $<TARGET_FILE:${PROJECT_NAME}> &&
            echo "" && echo "4 Threads:" &&
            ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=4 $<TARGET_FILE:${PROJECT_NAME}> &&
            echo "" && echo "8 Threads:" &&
            ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=8 $<TARGET_FILE:${PROJECT_NAME}>
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running comprehensive benchmarks"
)

# C++23 support test
add_custom_target(test_cpp23
    COMMAND ${CMAKE_CXX_COMPILER} -std=c++23 -dM -E -x c++ - < NUL | findstr "__cplusplus"
    COMMENT "Testing C++23 support"
)

# OpenMP test
add_custom_target(test_openmp
    COMMAND echo "int main(){return 0;}" | ${CMAKE_CXX_COMPILER} -std=c++23 -fopenmp -o test_temp -x c++ - && echo "OpenMP support OK" && del test_temp.exe
    COMMENT "Testing OpenMP support"
)

# Print configuration information
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP found: ${OpenMP_FOUND}")
if(OpenMP_FOUND)
    message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
endif()
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")
