#include <mannumopt/gauss-newton.hpp>

#include "pymannumopt.hpp"
#include "line-search.hpp"

namespace pymannumopt {
typedef mannumopt::GaussNewton<double, Eigen::Dynamic, Eigen::Dynamic> GaussNewton;

py::tuple call_minimize(GaussNewton& algo, VectorFunction& cost, VectorXd x0, lineSearch::Base& ls, py::object integrate, std::string decomposition)
{
  static const std::map<std::string, short> decompositions {
    { "LDLT", 0},
    { "FullPivHouseholderQR", 3},
    { "SVD", 4},
  };

  if (decompositions.count(decomposition) == 0) {
    std::stringstream ss;
    ss << "Decomposition must be one of (";
    for (const auto& pair : decompositions) {
      ss << pair.first << ", ";
    }
    ss << ").";
    throw std::invalid_argument(ss.str());
  }

  bool res;
  if (integrate.is(py::none())) {
    switch (decompositions.at(decomposition)) {
      case 0: res = algo.minimize<lineSearch::Base, VectorFunction, Eigen::LDLT<GaussNewton::MatrixTT>>(cost, x0, ls); break;
      case 3: res = algo.minimize<lineSearch::Base, VectorFunction, Eigen::FullPivHouseholderQR<GaussNewton::MatrixTT>>(cost, x0, ls); break;
      case 4: res = algo.minimize<lineSearch::Base, VectorFunction, Eigen::JacobiSVD<GaussNewton::MatrixTT>>(cost, x0, ls); break;
    }
  }
  else {
    switch (decompositions.at(decomposition)) {
      case 0: res = algo.minimize<lineSearch::Base, VectorFunction, integrate_type, Eigen::LDLT<GaussNewton::MatrixTT>>(cost, make_integrate(integrate), x0, ls); break;
      case 3: res = algo.minimize<lineSearch::Base, VectorFunction, integrate_type, Eigen::FullPivHouseholderQR<GaussNewton::MatrixTT>>(cost, make_integrate(integrate), x0, ls); break;
      case 4: res = algo.minimize<lineSearch::Base, VectorFunction, integrate_type, Eigen::JacobiSVD<GaussNewton::MatrixTT>>(cost, make_integrate(integrate), x0, ls); break;
    }
  }
  return py::make_tuple(res, x0);
}

void exposeGaussNewton(py::module_ m)
{
  using namespace pybind11::literals;

  py::class_<GaussNewton, Algo>(m, "GaussNewton")
    .def(py::init<int, int>())
    .def(py::init<int>())
    .def("minimize", &call_minimize, "cost"_a, "x0"_a, "line_search"_a = lineSearch::Default(), "integrate"_a = py::none(), "decomposition"_a = "LDLT")
    ;
}

} // namespace pymannumopt
