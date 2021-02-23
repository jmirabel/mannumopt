add_subdirectory(pybind11)

pybind11_add_module(pymannumopt
  pymannumopt.hpp
  pymannumopt.cpp
  numdiff.cpp
  line-search.hpp
  line-search.cpp
  bfgs.cpp
  newton.cpp
  augmented-lagrangian.cpp
  )
target_link_libraries(pymannumopt PUBLIC mannumopt)
set_target_properties(pymannumopt
  PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/py${PROJECT_NAME}")
file(COPY __init__.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/py${PROJECT_NAME}")

function(ADD_PYTHON_UNIT_TEST NAME SOURCE)
  add_test(NAME ${NAME} COMMAND ${PYTHON_EXECUTABLE} "${PROJECT_SOURCE_DIR}/${SOURCE}")
  set(PYTHONPATH "${CMAKE_CURRENT_BINARY_DIR}")

  IF(DEFINED ENV{PYTHONPATH})
    list(APPEND PYTHONPATH "$ENV{PYTHONPATH}")
  ENDIF(DEFINED ENV{PYTHONPATH})

  IF(WIN32)
    string(REPLACE ";" ":" PYTHONPATH_STR "${PYTHONPATH}")
  ELSE(WIN32)
    # get path separator to join those paths
    execute_process(COMMAND
      "${PYTHON_EXECUTABLE}" "-c" "import os; print(os.pathsep)"
      OUTPUT_VARIABLE PATHSEP
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE ";" "${PATHSEP}" PYTHONPATH_STR "${PYTHONPATH}")
  ENDIF(WIN32)
  set(ENV_VARIABLES "PYTHONPATH=${PYTHONPATH_STR}")
  IF(APPLE)
    list(APPEND ENV_VARIABLES "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
    list(APPEND ENV_VARIABLES "DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}")
  ENDIF(APPLE)
  set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "${ENV_VARIABLES}")
endfunction(ADD_PYTHON_UNIT_TEST NAME SOURCE)

ADD_PYTHON_UNIT_TEST(test-python-function tests/python/function.py)

if(NOT DEFINED PYTHON_SITELIB)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c"
    #"import sys, os; print(os.sep.join(['lib', 'python' + sys.version[:3], 'dist-packages']))"
    "from distutils import sysconfig; print(sysconfig.get_python_lib(prefix='', plat_specific=False))"
    OUTPUT_VARIABLE PYTHON_SITELIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

message(STATUS "Installing python module to ${PYTHON_SITELIB}")
install(TARGETS pymannumopt
  DESTINATION ${PYTHON_SITELIB}/pymannumopt)
install(FILES __init__.py
  DESTINATION ${PYTHON_SITELIB}/pymannumopt)
