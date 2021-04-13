#include <mannumopt/gauss-newton.hpp>
#include <mannumopt/line-search/armijo.hpp>

#include <iostream>
#include <boost/test/unit_test.hpp>

#include "test-algo-utils.hpp"

namespace chrono = std::chrono;

template<typename Algo, typename Time, typename Func, typename Xtype>
void gn_status(std::string what, Algo algo, bool success, Time start, Time end, Func f, Xtype x)
{
  double v;
  f.residual(x,v);
  status(what, algo, success, start, end, f, x, v);
  BOOST_TEST_MESSAGE("Expected solution: " << f.expectedCoeffs.transpose());
}

template<int N, int M>
auto polynom(const Eigen::Matrix<double, N, 1>& coeffs, const Eigen::Matrix<double, M, 1>& x)
  -> Eigen::Matrix<double, M, 1>
{
  Eigen::Matrix<double, M, 1> res (coeffs[0] + coeffs[1] * x.array());
  Eigen::Array<double, M, 1> x_pow_i (x);
  for (int i = 2; i < coeffs.size(); ++i) {
    x_pow_i *= x.array();
    res.array() += x_pow_i * coeffs[i];
  }
  return res;
}

template<int N, int M>
auto dpolynom(const Eigen::Matrix<double, N, 1>& coeffs, const Eigen::Matrix<double, M, 1>& x)
  -> Eigen::Matrix<double, M, N>
{
  Eigen::Matrix<double, M, N> res (x.size(), coeffs.size());
  res.col(0).setOnes();
  res.col(1) = x;
  for (int i = 2; i < coeffs.size(); ++i)
    res.col(i) = res.col(i-1).array() * x.array();
  return res;
}

template<int N, int M>
auto expression(const Eigen::Matrix<double, N, 1>& coeffs, const Eigen::Matrix<double, M, 1>& x)
  -> Eigen::Matrix<double, M, 1>
{
  static_assert(N==Eigen::Dynamic || N%2 == 1, "N should be uneven");
  assert(coeffs.size()%2==1);
  Eigen::Matrix<double, M, 1> res(x.size());
  res.setConstant (coeffs[0]);
  for (int k = 1; k < coeffs.size(); k+=2)
    res.array() += coeffs[k] * (x.array()*coeffs[k+1]).exp();
  return res;
}

template<int N, int M>
auto dexpression(const Eigen::Matrix<double, N, 1>& coeffs, const Eigen::Matrix<double, M, 1>& x)
  -> Eigen::Matrix<double, M, N>
{
  static_assert(N==Eigen::Dynamic || N%2 == 1, "N should be uneven");
  assert(coeffs.size()%2==1);
  Eigen::Matrix<double, M, N> res (x.size(), coeffs.size());
  res.col(0).setOnes();
  for (int k = 1; k < coeffs.size(); k+=2) {
    res.col(k) = (x.array()*coeffs[k+1]).exp();
    res.col(k+1) = coeffs[k] * x.array() * res.col(k).array();
  }
  return res;
}

enum FunctionType {
  PolynomFitting,
  ComplexExpression,
};

/// Find expression that fits the data
/// \tparam N + 1: order of the polynom
/// \tparam M number of measurements
template<int N, int M>
struct ExpressionFitting : mannumopt::VectorFunction<double, M, N> {
  MANNUMOPT_EIGEN_TYPEDEFS(double, N, S);

  typedef Eigen::Matrix<double, M, 1> ValueType;
  typedef Eigen::Matrix<double, M, N> DerivativeType;

  using ExpressionValue = ValueType (*)(const VectorS&, const ValueType&);
  using ExpressionDerivative = DerivativeType (*)(const VectorS&, const ValueType&);

  ExpressionValue exprV;
  ExpressionDerivative exprD;
  VectorS expectedCoeffs;
  ValueType inputs, outputs;

  constexpr int dimension() { return inputs.size(); }

  void f(const VectorS& X, ValueType& f) override
  {
    f = exprV(X, inputs) - outputs;
  }

  void f_fx(const VectorS& X, ValueType& f, DerivativeType& fx) override
  {
    this->f(X, f);
    fx = exprD(X, inputs);
  }

  ExpressionFitting(FunctionType type, const VectorS& coeffs, double err, int m = M) : inputs(m) {
    switch (type) {
      case PolynomFitting:
        exprV = &polynom<N,M>;
        exprD = &dpolynom<N,M>;
        break;
      case ComplexExpression:
        exprV = &expression<N,M>;
        exprD = &dexpression<N,M>;
        break;
    }
    expectedCoeffs = coeffs;
    inputs.setRandom();
    outputs = exprV(coeffs, inputs) + err * ValueType::Random(m);
  }
};

template<template<class,int,int> class LineSearch, int N, int M, template<int, int> class Function>
void gauss_newton(const char* type, Function<N, M> func, typename Function<N, M>::VectorS x)
{
  mannumopt::GaussNewton<double, N> gn(x.size());
  gn.xtol = 1e-8;
  gn.fxtol2 = 1e-12;
  gn.maxIter = 100;
  if (verbosityLevel() > 0)
    gn.cout = &std::cout;

  auto start = chrono::steady_clock::now();
  bool res;
  try {
    res = gn.template minimize<LineSearch<double, N, N>>(func, x);
  } catch (const std::runtime_error& e) {
    BOOST_TEST_MESSAGE("Caught std::runtime_error: " << e.what());
    res = false;
  }
  auto end = chrono::steady_clock::now();
  gn_status(type, gn, res, start, end, func, x);
}

template<int N, int M> void test_gauss_newton_tpl(int n, int m, FunctionType type, double err)
{
  Eigen::Matrix<double, N, 1> x(n), coeffs(n);
  BOOST_TEST_MESSAGE("noise level: " << err);
  for (int i = 0; i < 10; ++i) {
    x.setRandom();
    x.array() += 1.;

    coeffs.setRandom();
    coeffs = 0.5 * coeffs.array() + 1.5;

    ExpressionFitting<N, M> func(type, coeffs, err, m);

    gauss_newton<mannumopt::lineSearch::Armijo>("gauss_newton-armijo", func, x);
    gauss_newton<mannumopt::lineSearch::BisectionWeakWolfe>("gauss_newton-bisection", func, x);
  }
}

template<int N, int M> void test_gauss_newton(FunctionType type, double err)
{
  // Size known at compile time
  test_gauss_newton_tpl<N, M>(N, M, type, err);
  // Size known at run time
  test_gauss_newton_tpl<Eigen::Dynamic, Eigen::Dynamic>(N, M, type, err);
}

template<int N, int M> void test_gauss_newton(FunctionType type)
{
  test_gauss_newton<N, M>(type, 0);
  test_gauss_newton<N, M>(type, 0.02);
}

BOOST_AUTO_TEST_SUITE(TestPolynomFitting)
_SINGLE_TEST(gauss_newton, false, 3, 10, PolynomFitting)
_SINGLE_TEST(gauss_newton, false, 5, 20, PolynomFitting)
_SINGLE_TEST(gauss_newton, false, 7, 99, PolynomFitting)
BOOST_AUTO_TEST_SUITE_END()
 
BOOST_AUTO_TEST_SUITE(TestComplexExpression)
_SINGLE_TEST(gauss_newton, true, 3, 10, ComplexExpression)
_SINGLE_TEST(gauss_newton, true, 5, 20, ComplexExpression)
_SINGLE_TEST(gauss_newton, true, 7, 99, ComplexExpression)
BOOST_AUTO_TEST_SUITE_END()
