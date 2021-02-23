#include <mannumopt/newton.hpp>
#include <mannumopt/line-search/armijo.hpp>

#include <boost/test/unit_test.hpp>

#include "test-algo-utils.hpp"

namespace chrono = std::chrono;

template<template<class,int,int> class LineSearch, int N, template<int> class Function>
void newtonLS(const char* type, Function<N> func, typename Function<N>::VectorS x)
{
  mannumopt::NewtonLS<double, N> newton (x.size());
  newton.fxtol2 = 1e-12;
  newton.maxIter = 100;
  if (verbosityLevel() > 0)
    newton.cout = &std::cout;

  auto start = chrono::steady_clock::now();
  bool res;
  try {
    res = newton.minimize(func, x, LineSearch<double, N, N>());
  } catch (const std::runtime_error& e) {
    BOOST_TEST_MESSAGE("Caught std::runtime_error: " << e.what());
    res = false;
  }
  auto end = chrono::steady_clock::now();
  status(type, newton, res, start, end, func, x);
}

template<int N, template<int> class Function, class... Args> void test_newtonLS_tpl(int n, Args&&... args)
{
  Eigen::Matrix<double, N, 1> x(n);
  for (int i = 0; i < 20; ++i) {
    x.setRandom();
    x.array() += 1.;
    x *= 10;

    Function<N> func(std::forward<Args>(args)...);

    newtonLS<mannumopt::lineSearch::Armijo>("newton-ls-armijo", func, x);
    newtonLS<mannumopt::lineSearch::BisectionWeakWolfe>("newton-ls-bisection", func, x);
  }
}

template<int N, template<int> class Function, class... Args> void test_newtonLS(Args&&... args)
{
  // Size known at compile time
  test_newtonLS_tpl<N, Function, Args...>(N, std::forward<Args>(args)...);
  // Size known at run time
  test_newtonLS_tpl<Eigen::Dynamic, Function, Args...>(N, std::forward<Args>(args)...);
}

TEST_ALGO(newtonLS)
