#include <mannumopt/newton.hpp>

#include "pymannumopt.hpp"
#include "line-search.hpp"

namespace pymannumopt {

typedef mannumopt::NewtonLS<double, Eigen::Dynamic, Eigen::Dynamic> NewtonLS;

py::tuple call_minimize(NewtonLS& algo, Function& cost, VectorXd x0, lineSearch::Base& ls, py::object integrate)
{
  bool res;
  if (integrate.is(py::none()))
    res = algo.minimize(cost, x0, ls);
  else
    res = algo.minimize(cost, make_integrate(integrate), x0, ls);
  return py::make_tuple(res, x0);
}

void exposeNewton(py::module_ m)
{
  using namespace pybind11::literals;

  py::class_<NewtonLS, Algo>(m, "NewtonLS")
    .def(py::init<int, int>())
    .def(py::init<int>())
    .def("minimize", &call_minimize, "cost"_a, "x0"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())

    .def_readwrite("eta", &NewtonLS::eta)
    .def_readwrite("u_maxstep", &NewtonLS::u_maxstep)
    ;
}

} // namespace pymannumopt
