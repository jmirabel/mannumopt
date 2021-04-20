cmake_minimum_required(VERSION 3.10)

project(mannumopt
  VERSION 0.0.0
  LANGUAGES CXX)

install(DIRECTORY include/mannumopt
  DESTINATION include/mannumopt
  FILES_MATCHING PATTERN "*.hpp")

find_package(Eigen3 3.3 REQUIRED)
add_library(mannumopt INTERFACE)
target_include_directories(mannumopt INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mannumopt INTERFACE Eigen3::Eigen)

set(MANNUMOPT_PYTHON_BINDINGS ON CACHE BOOL "Enable Python bindings")
if (MANNUMOPT_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

enable_testing()
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
