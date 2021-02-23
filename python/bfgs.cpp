#include <mannumopt/bfgs.hpp>

#include "pymannumopt.hpp"
#include "line-search.hpp"

namespace pymannumopt {
typedef mannumopt::BFGS<double, Eigen::Dynamic, Eigen::Dynamic> BFGS;

py::tuple call_minimize(BFGS& algo, Function& cost, VectorXd x0, lineSearch::Base& ls, py::object integrate)
{
  bool res;
  if (integrate.is(py::none()))
    res = algo.minimize(cost, x0, ls);
  else
    res = algo.minimize(cost, make_integrate(integrate), x0, ls);
  return py::make_tuple(res, x0);
}

void exposeBFGS(py::module_ m)
{
  using namespace pybind11::literals;

  py::class_<BFGS, Algo>(m, "BFGS")
    .def(py::init<int, int>())
    .def(py::init<int>())
    .def("minimize", &call_minimize, "cost"_a, "x0"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())

    .def_readwrite("fxx_i", &BFGS::fxx_i)
    ;
}

} // namespace pymannumopt
