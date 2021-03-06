#include <mannumopt/penalty.hpp>
#include <mannumopt/bfgs.hpp>
#include <mannumopt/line-search/armijo.hpp>

#include <boost/test/unit_test.hpp>

#include "test-algo-utils.hpp"

namespace chrono = std::chrono;

template<template<class,int,int> class LineSearch, int N, template<int> class CostFunction,
  class EConstraint>
void penalty(const char* type, CostFunction<N> cost,
    EConstraint econstraint, typename CostFunction<N>::VectorS x)
{
  mannumopt::Penalty<double, N, N, EConstraint::dimension()> al(x.size(), x.size(), econstraint.dimension());
  al.etol2 = 1e-12;
  al.fxtol2 = 1e-10;
  al.maxIter = 40;
  if (verbosityLevel() > 0)
    al.cout = &std::cout;

  mannumopt::BFGS<double, N> bfgs(x.size());
  bfgs.maxIter = 100;
  if (verbosityLevel() > 1)
    bfgs.cout = &std::cout;

  bfgs.fxx_i.setIdentity();
  auto start = chrono::steady_clock::now();
  bool res;
  try {
    res = al.template minimize<LineSearch<double,N,N>>(cost, econstraint, x, bfgs);
  } catch (const std::runtime_error& e) {
    BOOST_TEST_MESSAGE("Caught std::runtime_error: " << e.what());
    res = false;
  }
  auto end = chrono::steady_clock::now();
  status(type, al, res, start, end, cost, x);
}

template<typename ScalarFunctor, int N>
struct ScalarToVector : ScalarFunctor {
  MANNUMOPT_EIGEN_TYPEDEFS(double, N, S);

  typedef Eigen::Matrix<double, 1, 1> Vector;
  typedef Eigen::Matrix<double, 1, N> Matrix;

  using ScalarFunctor::ScalarFunctor;

  static constexpr int dimension() { return 1; }

  void f(const VectorS& X, Vector& f)
  {
    ScalarFunctor::f(X, f(0));
  }

  void f_fx(const VectorS& X, Vector& f, Matrix& fx)
  {
    ScalarFunctor::f_fx(X, f(0), fx);
  }
};

template<int N, template<int> class CostFunction, class... Args> void test_penalty_tpl(int n, Args&&... args)
{
  Eigen::Matrix<double, N, 1> x(n);
  ScalarToVector< Quadratic<N>, N> ec(n);
  ec.A = decltype(ec.A)::Identity(n,n);
  ec.B = decltype(ec.B)::Zero(n);
  ec.c = -1;

  for (int i = 0; i < 20; ++i) {
    x.setRandom();
    x.array() += 1.;
    x *= 10;

    CostFunction<N> cost(std::forward<Args>(args)...);

    penalty<mannumopt::lineSearch::Armijo>("al-bfgs-armijo", cost, ec, x);
    penalty<mannumopt::lineSearch::BisectionWeakWolfe>("al-bfgs-bisection", cost, ec, x);
  }
}

template<int N, template<int> class CostFunction, class... Args> void test_penalty(Args&&... args)
{
  // Size known at compile time
  test_penalty_tpl<N, CostFunction, Args...>(N, std::forward<Args>(args)...);
  // Size known at run time
  test_penalty_tpl<Eigen::Dynamic, CostFunction, Args...>(N, std::forward<Args>(args)...);
}

//TEST_ALGO(penalty)
_SINGLE_TEST(penalty,false,4,Rosenbrock, 1.)
_SINGLE_TEST(penalty,false,6,Rosenbrock, 1.)
_SINGLE_TEST(penalty,false,8,Rosenbrock, 1.)
