cmake_minimum_required(VERSION 3.12)
project(svd_eigen)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(svd_eigen MODULE svd_eigen.cpp)
target_link_libraries(svd_eigen PRIVATE pybind11::module Eigen3::Eigen)
set_target_properties(svd_eigen PROPERTIES CXX_STANDARD 14)