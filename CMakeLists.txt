cmake_minimum_required(VERSION 3.10)

project(mannumopt
  VERSION 0.0.0
  LANGUAGES CXX)

install(DIRECTORY include/mannumopt
  DESTINATION include/mannumopt)

find_package(Eigen3 3.3 REQUIRED)
add_library(mannumopt INTERFACE)
target_include_directories(mannumopt INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mannumopt INTERFACE Eigen3::Eigen)

enable_testing()
add_subdirectory(tests)
