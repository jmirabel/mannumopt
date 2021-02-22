#include <mannumopt/bfgs.hpp>
#include <mannumopt/line-search/armijo.hpp>

#include <boost/test/unit_test.hpp>

#include "test-algo-utils.hpp"

namespace chrono = std::chrono;

template<template<class,int,int> class LineSearch, int N, template<int> class Function>
void bfgs(const char* type, Function<N> func, typename Function<N>::VectorS x)
{
  mannumopt::BFGS<double, N> bfgs(x.size(), x.size());
  bfgs.fxtol2 = 1e-12;
  bfgs.maxIter = 100;
  //bfgs.cout = &std::cout;

  bfgs.fxx_i.setIdentity();
  auto start = chrono::steady_clock::now();
  bool res;
  try {
    res = bfgs.minimize(func, x, LineSearch<double, N, N>());
  } catch (const std::runtime_error& e) {
    BOOST_TEST_MESSAGE("Caught std::runtime_error: " << e.what());
    res = false;
  }
  auto end = chrono::steady_clock::now();
  status(type, bfgs, res, start, end, func, x);
}

template<int N, template<int> class Function, class... Args> void test_bfgs(Args&&... args)
{
  { // Size known at compile time
    Eigen::Matrix<double, N, 1> x;
    for (int i = 0; i < 10; ++i) {
      x.setRandom();
      x.array() += 1.;
      x *= 10;

      Function<N> func(std::forward<Args>(args)...);

      bfgs<mannumopt::lineSearch::Armijo>("bfgs-armijo", func, x);
      bfgs<mannumopt::lineSearch::BisectionWeakWolfe>("bfgs-bisection", func, x);
      //slsqp_nlopt(func, x);
    }
  }

  { // Size known at run time
    Eigen::Matrix<double, Eigen::Dynamic, 1> x(N);
    for (int i = 0; i < 10; ++i) {
      x.setRandom();
      x.array() += 1.;
      x *= 10;

      Function<Eigen::Dynamic> func(std::forward<Args>(args)...);

      bfgs<mannumopt::lineSearch::Armijo>("bfgs-armijo", func, x);
      bfgs<mannumopt::lineSearch::BisectionWeakWolfe>("bfgs-bisection", func, x);
      //slsqp_nlopt(func, x);
    }
  }
}

TEST_ALGO(bfgs)
