#include <mannumopt/numdiff.hpp>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include "test-algo-utils.hpp"

using namespace mannumopt;
using Eigen::VectorXd;

template<typename Function>
struct ScalarToVector : VectorFunction<double, 1, Eigen::Dynamic, Eigen::Dynamic> {
  typedef VectorFunction<double, 1, Eigen::Dynamic, Eigen::Dynamic> Base;
  using Base::VectorX;
  using Base::VectorN;
  using Base::MatrixNT;

  Function func;

  ScalarToVector(Function f) : func(f) {}

  int dimension() override { return 1.; }

  void f(const VectorX& X, VectorN& f) override
  {
    func.f(X, f(0));
  }

  void f_fx(const VectorX& X, VectorN& f, MatrixNT& fx) override
  {
    func.f_fx(X, f(0), fx);
  }
};

template<typename Function>
void test_num_diff(Function func, double eps, const VectorXd& X,
    int outputSize)
{
  typename Function::Derivative
    fx_for(outputSize, X.size()),
    fx_cen(outputSize, X.size()),
    fx_ana(outputSize, X.size());

  typename Function::Output f;

  numdiff::forward(func, X, eps, fx_for);
  numdiff::central(func, X, eps, fx_cen);

  func.f_fx(X, f, fx_ana);

  BOOST_CHECK(fx_for.isApprox(fx_ana, std::sqrt(eps)));
  BOOST_CHECK(fx_cen.isApprox(fx_ana, eps));
}

template<typename IFunction>
void test_vector_num_diff(IFunction ifunc, double eps, const VectorXd& X,
    int outputSize)
{
  typedef ScalarToVector<IFunction> Function;
  Function func(ifunc);

  test_num_diff(func, eps, X, outputSize);
}

template<typename Function>
void test_second_order_num_diff(Function func, double eps, const VectorXd& X,
    int outputSize)
{
  typename Function::Hessian
    fxx_cen(X.size(), X.size()),
    fxx_ana(X.size(), X.size());

  typename Function::Output f;
  typename Function::Derivative fx (X.size());

  numdiff::second_order_central(func, X, eps, fxx_cen);

  func.f_fx_fxx(X, f, fx, fxx_ana);

  BOOST_CHECK(fxx_cen.isApprox(fxx_ana, eps));
}

BOOST_AUTO_TEST_CASE(rosenbrock)
{
  Rosenbrock<Eigen::Dynamic> rosenbrock(1.);
  test_num_diff(rosenbrock, 1e-5, VectorXd::Random(3), 1);
  test_vector_num_diff(rosenbrock, 1e-5, VectorXd::Random(3), 1);
  test_second_order_num_diff(rosenbrock, 1e-5, VectorXd::Random(3), 1);
}

BOOST_AUTO_TEST_CASE(quadratic)
{
  Quadratic<Eigen::Dynamic> quad(VectorXd::Random(3).asDiagonal());
  test_num_diff(quad, 1e-5, VectorXd::Random(3), 1);
  test_vector_num_diff(quad, 1e-5, VectorXd::Random(3), 1);
  test_second_order_num_diff(quad, 1e-5, VectorXd::Random(3), 1);
}
