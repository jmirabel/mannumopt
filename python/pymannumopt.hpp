#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>

#include <mannumopt/fwd.hpp>
#include <mannumopt/function.hpp>

namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::MatrixXd;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXd;

namespace pymannumopt {
typedef mannumopt::Algo<double, Eigen::Dynamic> Algo;

typedef mannumopt::Function<double, Eigen::Dynamic, Eigen::Dynamic> FunctionBase;
typedef mannumopt::VectorFunction<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> VectorFunctionBase;

struct Function : FunctionBase
{
  virtual ~Function() = default;
  virtual double f_py(const VectorXd& X) = 0;
  virtual double f_fx_py(const VectorXd& X, Eigen::Ref<RowVectorXd> fx) = 0;
  virtual double f_fx_fxx_py(const VectorXd& X, Eigen::Ref<RowVectorXd> fx, Eigen::Ref<MatrixXd> fxx)
  {
    throw std::logic_error("pymannumopt::Function::f_fx_fxx not implemented");
  }

  void f(const VectorXd& X, double& f) override
  {
    f = f_py(X);
  }

  void f_fx(const VectorXd& X, double& f, RowVectorXd& fx) override
  {
    f = f_fx_py(X, fx);
  }

  void f_fx_fxx(const VectorXd& X, double& f, RowVectorXd& fx, MatrixXd& fxx) override
  {
    f = f_fx_fxx_py(X, fx, fxx);
  }
};

struct VectorFunction : VectorFunctionBase {
  virtual void f_py(const VectorXd& X, Eigen::Ref<VectorXd> f) = 0;
  virtual void f_fx_py(const VectorXd& X, Eigen::Ref<VectorXd> f, Eigen::Ref<MatrixXd> fx) = 0;

  void f(const VectorXd& X, VectorXd& f) override
  {
    f_py(X, f);
  }

  void f_fx(const VectorXd& X, VectorXd& f, MatrixXd& fx) override
  {
    f_fx_py(X, f, fx);
  }
};

typedef std::function<void (VectorXd& x_plus_v, const VectorXd& x, const VectorXd& v)> integrate_type;
inline integrate_type make_integrate(py::object func)
{
  return [&func](VectorXd& x_plus_v, const VectorXd& x, const VectorXd& v) { x_plus_v = func(x, v).cast<VectorXd>(); };
}

void exposeNumDiff(py::module_ m);
void exposeLineSearch(py::module_ m);
void exposeBFGS(py::module_ m);
void exposeNewton(py::module_ m);
void exposeAugmentedLagrangian(py::module_ m);
} // namespace pymannumopt

