add_subdirectory(pybind11)

pybind11_add_module(pymannumopt
  pymannumopt.hpp
  pymannumopt.cpp
  numdiff.cpp
  line-search.hpp
  line-search.cpp
  bfgs.cpp
  augmented-lagrangian.cpp
  )
target_link_libraries(pymannumopt PUBLIC mannumopt)

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
