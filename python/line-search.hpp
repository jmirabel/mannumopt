#pragma once

#include "pymannumopt.hpp"

#include <mannumopt/line-search/armijo.hpp>
#include <mannumopt/function.hpp>

namespace pymannumopt {
namespace lineSearch {

struct Base
{
virtual ~Base() = default;
virtual void operator() (FunctionBase& func, integrate_type integrate,
    const VectorXd& x, const VectorXd& p, double f, const RowVectorXd& fx,
    double& a, VectorXd& x2) = 0;
};

struct Armijo : Base
{
  mannumopt::lineSearch::Armijo<double, Eigen::Dynamic> ls;

  void operator() (FunctionBase& func, integrate_type integrate,
      const VectorXd& x, const VectorXd& p, double f, const RowVectorXd& fx,
      double& a, VectorXd& x2)
  {
    ls(func, integrate, x, p, f, fx, a, x2);
  }
};

struct BisectionWeakWolfe : Base
{
  mannumopt::lineSearch::BisectionWeakWolfe<double, Eigen::Dynamic> ls;

  void operator() (FunctionBase& func, integrate_type integrate,
      const VectorXd& x, const VectorXd& p, double f, const RowVectorXd& fx,
      double& a, VectorXd& x2)
  {
    ls(func, integrate, x, p, f, fx, a, x2);
  }
};

typedef Armijo Default;
} // namespace lineSearch
} // namespace pymannumopt
