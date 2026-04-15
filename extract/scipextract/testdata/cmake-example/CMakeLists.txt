cmake_minimum_required(VERSION 3.15)
project(cmake_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(cmake_math src/math.cpp)
add_executable(cmake_example src/main.cpp)
target_link_libraries(cmake_example PRIVATE cmake_math)
