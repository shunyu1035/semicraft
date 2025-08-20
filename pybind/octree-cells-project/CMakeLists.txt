cmake_minimum_required(VERSION 3.10)

project(OctreeCellsProject)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(src)

file(GLOB SOURCES
    src/*.cpp
    src/types/*.h
)

add_executable(OctreeCells ${SOURCES})