#include <mannumopt/numdiff.hpp>
#include "pymannumopt.hpp"

#include <pybind11/functional.h>

namespace pymannumopt {

#define EXPOSE_ND(method, FuncType, MatrixType)                                \
  m.def(#method, [](FuncType& func, const VectorXd& X, double eps,             \
      Eigen::Ref<MatrixType> fx, py::object integrate)                         \
      {                                                                        \
        if (integrate.is(py::none()))                                          \
          numdiff::method(func, X, eps, fx);                                   \
        else                                                                   \
          numdiff::method(func, X, eps, fx, make_integrate(integrate));        \
      }, "function"_a, "x"_a, "eps"_a, "fxx"_a, "integrate"_a = py::none())

void exposeNumDiff(py::module_ main)
{
  using namespace mannumopt;
  using namespace pybind11::literals;

  auto m = main.def_submodule("numdiff");
  EXPOSE_ND(forward, Function, RowVectorXd);
  EXPOSE_ND(forward, VectorFunction, MatrixXd);
  EXPOSE_ND(forward, VectorFunction, RowMajorMatrixXd);
  EXPOSE_ND(central, Function, RowVectorXd);
  EXPOSE_ND(central, VectorFunction, MatrixXd);
  EXPOSE_ND(central, VectorFunction, RowMajorMatrixXd);
  EXPOSE_ND(second_order_central, Function, MatrixXd);
  EXPOSE_ND(second_order_central, Function, RowMajorMatrixXd);
}

} // namespace pymannumopt
