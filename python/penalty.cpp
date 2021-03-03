#include <mannumopt/penalty.hpp>
#include <mannumopt/bfgs.hpp>
#include <mannumopt/newton.hpp>
#include <mannumopt/gauss-newton.hpp>

#include "pymannumopt.hpp"
#include "line-search.hpp"

namespace pymannumopt {

typedef mannumopt::Penalty<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> Penalty;
typedef mannumopt::BFGS<double, Eigen::Dynamic, Eigen::Dynamic> BFGS;
typedef mannumopt::NewtonLS<double, Eigen::Dynamic, Eigen::Dynamic> NewtonLS;
typedef mannumopt::GaussNewton<double, Eigen::Dynamic, Eigen::Dynamic> GaussNewton;

template<typename InnerAlgo, typename Func>
py::tuple call_minimize(Penalty& pen, Func& cost, VectorFunction& equality, VectorXd x0, InnerAlgo& ialgo, lineSearch::Base& ls, py::object integrate)
{
  bool res;
  if (integrate.is(py::none()))
    res = pen.minimize(cost, equality, x0, ialgo, ls);
  else
    res = pen.minimize(cost, equality, make_integrate(integrate), x0, ialgo, ls);
  return py::make_tuple(res, x0);
}

void exposePenalty(py::module_ m)
{
  using namespace pybind11::literals;

  py::class_<Penalty, Algo>(m, "Penalty")
    .def(py::init<int, int, int>())
    .def(py::init<int, int>())
    .def_readwrite("etol2", &Penalty::etol2)

    .def("minimize", &call_minimize<BFGS, Function>, "cost"_a, "constraints"_a, "x0"_a, "inner_algo"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())
    .def("minimize", &call_minimize<NewtonLS, Function>, "cost"_a, "constraints"_a, "x0"_a, "inner_algo"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())
    .def("minimize", &call_minimize<GaussNewton, VectorFunction>, "cost"_a, "constraints"_a, "x0"_a, "inner_algo"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none())
    ;
}

} // namespace pymannumopt
