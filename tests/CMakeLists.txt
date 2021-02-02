find_package(Boost COMPONENTS unit_test_framework REQUIRED)

function(add_cpp_test name)
  set(test_name test-cpp-${name})
  add_executable(${test_name} ${name}.cpp)
  target_link_libraries(${test_name} PUBLIC mannumopt Boost::unit_test_framework)
endfunction()

add_cpp_test(bfgs)
add_cpp_test(choleski)
