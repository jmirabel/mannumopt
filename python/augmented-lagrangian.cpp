#include <mannumopt/augmented-lagrangian.hpp>
#include <mannumopt/bfgs.hpp>
#include <mannumopt/newton.hpp>

#include "pymannumopt.hpp"
#include "line-search.hpp"

namespace pymannumopt {

typedef mannumopt::AugmentedLagrangian<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> AL;
typedef mannumopt::BFGS<double, Eigen::Dynamic, Eigen::Dynamic> BFGS;
typedef mannumopt::NewtonLS<double, Eigen::Dynamic, Eigen::Dynamic> NewtonLS;

template<typename InnerAlgo>
py::tuple call_minimize(AL& al, Function& cost, VectorFunction& equality, VectorXd x0, InnerAlgo& ialgo, lineSearch::Base& ls, py::object integrate)
{
  bool res;
  if (integrate.is(py::none()))
    res = al.minimize(cost, equality, x0, ialgo, ls);
  else
    res = al.minimize(cost, equality, make_integrate(integrate), x0, ialgo, ls);
  return py::make_tuple(res, x0);
}

void exposeAugmentedLagrangian(py::module_ m)
{
  using namespace pybind11::literals;

  py::class_<AL, Algo>(m, "AugmentedLagrangian")
    .def(py::init<int, int, int>())
    .def(py::init<int, int>())
    .def_readwrite("etol2", &AL::etol2)

    .def("minimize", &call_minimize<BFGS>, "cost"_a, "constraints"_a, "x0"_a, "inner_algo"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())
    .def("minimize", &call_minimize<NewtonLS>, "cost"_a, "constraints"_a, "x0"_a, "inner_algo"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())
    ;
}

} // namespace pymannumopt
