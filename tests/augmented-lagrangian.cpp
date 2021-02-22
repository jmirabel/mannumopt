#include <mannumopt/augmented-lagrangian.hpp>
#include <mannumopt/bfgs.hpp>
#include <mannumopt/line-search/armijo.hpp>

#include <boost/test/unit_test.hpp>

#include "test-algo-utils.hpp"

namespace chrono = std::chrono;

template<template<class,int,int> class LineSearch, int N, template<int> class CostFunction,
  class EConstraint>
void augmented_lagrangian(const char* type, CostFunction<N> cost,
    EConstraint econstraint, typename CostFunction<N>::VectorS x)
{
  mannumopt::AugmentedLagrangian<double, N, N, EConstraint::dimension()> al;
  al.etol2 = 1e-12;
  al.fxtol2 = 1e-10;
  al.maxIter = 40;
  //al.cout = &std::cout;

  mannumopt::BFGS<double, N> bfgs;
  bfgs.maxIter = 100;
  //bfgs.cout = &std::cout;

  bfgs.fxx_i.setIdentity();
  auto start = chrono::steady_clock::now();
  bool res;
  try {
    res = al.minimize(cost, econstraint, x, bfgs, LineSearch<double, N, N>());
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

template<int N, template<int> class CostFunction, class... Args> void test_augmented_lagrangian(Args&&... args)
{
  Eigen::Matrix<double, N, 1> x;
  ScalarToVector< Quadratic<N>, N> ec;
  ec.A.setIdentity();
  ec.B.setZero();
  ec.c = -1;

  for (int i = 0; i < 20; ++i) {
    x.setRandom();
    x.array() += 1.;
    x *= 10;

    CostFunction<N> cost(std::forward<Args>(args)...);

    augmented_lagrangian<mannumopt::lineSearch::Armijo>("al-bfgs-armijo", cost, ec, x);
    augmented_lagrangian<mannumopt::lineSearch::BisectionWeakWolfe>("al-bfgs-bisection", cost, ec, x);
  }
}

//TEST_ALGO(augmented_lagrangian)
_SINGLE_TEST(augmented_lagrangian,false,4,Rosenbrock, 1.)
_SINGLE_TEST(augmented_lagrangian,false,6,Rosenbrock, 1.)
_SINGLE_TEST(augmented_lagrangian,false,8,Rosenbrock, 1.)
