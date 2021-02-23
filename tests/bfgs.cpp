#include <mannumopt/bfgs.hpp>
#include <mannumopt/line-search/armijo.hpp>

#include <iostream>
#include <chrono>

#include <boost/test/unit_test.hpp>

#include "test-algo-utils.hpp"

namespace chrono = std::chrono;

template<template<class,int,int> class LineSearch, int N, template<int> class Function>
void bfgs(const char* type, Function<N> func, typename Function<N>::VectorS x)
{
  mannumopt::BFGS<double, N> bfgs(x.size(), x.size());
  bfgs.fxtol2 = 1e-12;
  bfgs.maxIter = 100;
  if (verbosityLevel() > 0)
    bfgs.cout = &std::cout;

  bfgs.fxx_i.setIdentity();
  auto start = chrono::steady_clock::now();
  bool res;
  try {
    res = bfgs.template minimize<LineSearch<double, N, N>>(func, x);
  } catch (const std::runtime_error& e) {
    BOOST_TEST_MESSAGE("Caught std::runtime_error: " << e.what());
    res = false;
  }
  auto end = chrono::steady_clock::now();
  status(type, bfgs, res, start, end, func, x);
}


template<int N, template<int> class Function, class... Args> void test_bfgs_tpl(int n, Args&&... args)
{
  Eigen::Matrix<double, N, 1> x(n);
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

template<int N, template<int> class Function, class... Args> void test_bfgs(Args&&... args)
{
  // Size known at compile time
  test_bfgs_tpl<N, Function, Args...>(N, std::forward<Args>(args)...);
  // Size known at run time
  test_bfgs_tpl<Eigen::Dynamic, Function, Args...>(N, std::forward<Args>(args)...);
}

TEST_ALGO(bfgs)
