#include <mannumopt/newton.hpp>
#include <mannumopt/trust-region/cauchy.hpp>
#include <mannumopt/trust-region/two-dimensional.hpp>

#include <chrono>

#include <boost/test/unit_test.hpp>

#include "test-algo-utils.hpp"

namespace chrono = std::chrono;

template<template<class,int> class TrustRegion, int N, template<int> class Function>
void newtonTR(const char* type, Function<N> func, typename Function<N>::VectorS x)
{
  mannumopt::NewtonTR<double, N> newton;
  newton.fxtol2 = 1e-12;
  newton.maxIter = 100;
  if (verbosityLevel() > 0)
    newton.cout = &std::cout;

  auto start = chrono::steady_clock::now();
  bool res;
  try {
    res = newton.minimize(func, x, TrustRegion<double, N>());
  } catch (const std::runtime_error& e) {
    BOOST_TEST_MESSAGE("Caught std::runtime_error: " << e.what());
    res = false;
  }
  auto end = chrono::steady_clock::now();
  status(type, newton, res, start, end, func, x);
}

template<int N, template<int> class Function, class... Args> void test_newtonTR(Args&&... args)
{
  Eigen::Matrix<double, N, 1> x;
  for (int i = 0; i < 20; ++i) {
    x.setRandom();
    x.array() += 1.;
    x *= 10;

    Function<N> func(std::forward<Args>(args)...);

    newtonTR<mannumopt::trustRegion::Cauchy>("newton-tr-cauchy", func, x);
    newtonTR<mannumopt::trustRegion::TwoDimensional>("newton-tr-2d", func, x);
  }
}

TEST_ALGO(newtonTR)
