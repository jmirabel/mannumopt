#include <mannumopt/bfgs.hpp>
#include <mannumopt/line-search/armijo.hpp>

#include <nlopt.hpp>

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

template<int N, template<int> class Function>
void slsqp_nlopt(Function<N> func, typename Function<N>::VectorS x)
{
  typedef typename Function<N>::VectorS VectorS;
  typedef typename Function<N>::RowVectorS RowVectorS;

  auto obj = [](const std::vector<double> &x, std::vector<double> &grad, void *data) -> double
  {
    Function<N>& func (*reinterpret_cast<Function<N>*>(data));

    VectorS X = Eigen::Map<const VectorS>(x.data());
    double f;
    if (grad.empty())
      func.f(X, f);
    else {
      RowVectorS fx;
      func.f_fx(X, f, fx);
      Eigen::Map<RowVectorS>(grad.data()) = fx;
    }
    return f;
  };


  nlopt::opt opt (nlopt::LD_SLSQP, N);
  //nlopt::opt opt (nlopt::LD_LBFGS, N);
  opt.set_min_objective(obj, &func);
  opt.set_stopval(0.000001);
  opt.set_maxeval(1000);

  std::vector<double> x0(N);
  Eigen::Map<VectorS>(x0.data()) = x;

  double minf;
  auto start = chrono::steady_clock::now(), end = start;

  bool res = false;
  try{
    start = chrono::steady_clock::now();
    nlopt::result result = opt.optimize(x0, minf);
    end = chrono::steady_clock::now();
    res = true;
  }
  catch(std::exception &e) {
    std::cout << "SLSQP failed: " << e.what() << std::endl;
  }
  x = Eigen::Map<const VectorS>(x0.data());
  struct { int iter = -1; } fake_algo;
  status("SLSQP", fake_algo, res, start, end, func, x);

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
