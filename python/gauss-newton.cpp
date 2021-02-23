#include <mannumopt/gauss-newton.hpp>

#include "pymannumopt.hpp"
#include "line-search.hpp"

namespace pymannumopt {
typedef mannumopt::GaussNewton<double, Eigen::Dynamic, Eigen::Dynamic> GaussNewton;

py::tuple call_minimize(GaussNewton& algo, VectorFunction& cost, VectorXd x0, lineSearch::Base& ls, py::object integrate)
{
  bool res;
  if (integrate.is(py::none()))
    res = algo.minimize(cost, x0, ls);
  else
    res = algo.minimize(cost, make_integrate(integrate), x0, ls);
  return py::make_tuple(res, x0);
}

void exposeGaussNewton(py::module_ m)
{
  using namespace pybind11::literals;

  py::class_<GaussNewton, Algo>(m, "GaussNewton")
    .def(py::init<int, int>())
    .def(py::init<int>())
    .def("minimize", &call_minimize, "cost"_a, "x0"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())
    ;
}

} // namespace pymannumopt
