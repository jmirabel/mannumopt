#include "pymannumopt.hpp"
#include <iostream>

namespace pymannumopt {

class PyFunction : public Function {
public:
  using Function::Function;
  virtual ~PyFunction() = default;

  virtual double f_py(const VectorXd& X) override {
    PYBIND11_OVERRIDE_PURE_NAME(double, Function, "f", f_py, X);
  }
  virtual double f_fx_py(const VectorXd& X, Eigen::Ref<RowVectorXd> fx) override {
    PYBIND11_OVERRIDE_PURE_NAME(double, Function, "f_fx", f_fx_py, X, fx);
  }
  virtual double f_fx_fxx_py(const VectorXd& X, Eigen::Ref<RowVectorXd> fx, Eigen::Ref<MatrixXd> fxx) override {
    PYBIND11_OVERRIDE_NAME(double, Function, "f_fx_fxx", f_fx_fxx_py, X, fx, fxx);
  }
};
class PyVectorFunction : public VectorFunction {
public:
  using VectorFunction::VectorFunction;
  virtual ~PyVectorFunction() = default;

  virtual int dimension() override {
    PYBIND11_OVERRIDE_PURE(int, VectorFunction, dimension);
  }

  virtual void f_py(const VectorXd& X, Eigen::Ref<VectorXd> f) override {
    PYBIND11_OVERRIDE_PURE_NAME(void, VectorFunction, "f", f_py, X, f);
  }
  virtual void f_fx_py(const VectorXd& X, Eigen::Ref<VectorXd> f, Eigen::Ref<MatrixXd> fx) override {
    PYBIND11_OVERRIDE_PURE_NAME(void, VectorFunction, "f_fx", f_fx_py, X, f, fx);
  }
};

} // namespace pymannumopt

PYBIND11_MODULE(pymannumopt, m) {
  using namespace pymannumopt;

  py::class_<Function, PyFunction>(m, "Function")
    .def(py::init<>())
    .def("f", &Function::f)
    .def("f_fx", &Function::f_fx)
    .def("f_fx_fxx", &Function::f_fx_fxx)
    ;
  py::class_<VectorFunction, PyVectorFunction>(m, "VectorFunction")
    .def(py::init<>())
    .def("f", &VectorFunction::f)
    .def("f_fx", &VectorFunction::f_fx)
    .def("eval_f", &VectorFunction::eval_f)
    ;

  py::class_<Algo>(m, "Algo")
    .def_property("verbose",
        &Algo::verbose,
        [](Algo& a, bool verbose) { a.cout = verbose ? &std::cout : nullptr; })
    .def_readwrite("xtol", &Algo::xtol)
    .def_readwrite("fxtol2", &Algo::fxtol2)
    .def_readwrite("maxIter", &Algo::maxIter)
    .def_readonly("iter", &Algo::iter)
    ;

  exposeNumDiff(m);
  exposeLineSearch(m);
  exposeBFGS(m);
  exposeNewton(m);
  exposeAugmentedLagrangian(m);
  exposeGaussNewton(m);
  exposePenalty(m);
}
