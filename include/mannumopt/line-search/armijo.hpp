#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt::lineSearch {
template<typename Scalar, int XDim, int TDim = XDim>
struct Armijo {
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, XDim, X);
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, TDim, T);

Scalar r = 0.5;
Scalar c = 0.1;
Scalar amin = 1e-8;

template<typename Functor, typename IntegrateFunctor>
void operator() (Functor& func, IntegrateFunctor integrate,
    const VectorX& x, const VectorT& p, Scalar f, const RowVectorT& fx,
    Scalar& a, VectorX& x2)
{
  Scalar m = fx * p;
  if (m >= 0) {
    std::stringstream ss;
    ss << "Not a valid descent direction: " << m << " should be < 0";
    throw std::runtime_error(ss.str());
  }
  Scalar f2;
  while(true) {
    integrate(x2, x, a*p);
    func.f(x2, f2);
    if (f2 < f + c * a * m) break;
    a *= r;
    if (a < amin) {
      std::stringstream ss;
      ss << "Not a valid descent direction: m (" << m << "). a (" << a << ") below amin (" << amin << ")";
      throw std::runtime_error(ss.str());
    }
  }
}
};

/// The algorithm comes from
/// https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf
template<typename Scalar, int XDim, int TDim = XDim>
struct BisectionWeakWolfe {
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, XDim, X);
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, TDim, T);

Scalar r = 0.5;
Scalar c1 = 0.1;
Scalar c2 = 0.9;
Scalar amin = 1e-8;

template<typename Functor, typename IntegrateFunctor>
void operator() (Functor& func, IntegrateFunctor integrate,
    const VectorX& x, const VectorT& p, Scalar f, const RowVectorT& fx,
    Scalar& a, VectorX& x2)
{
  Scalar m = fx * p;
  if (m >= 0)
    throw std::runtime_error("Not a valid descent direction");

  Scalar f2;
  RowVectorT fx2(fx.size());
  Scalar alpha = 0.;
  Scalar beta = -1.;

  a = 1.;
  int iter = 1000;
  while(true) {
    integrate(x2, x, a*p);
    func.f_fx(x2, f2, fx2);
    if (f2 > f + c1 * a * m) {
      beta = a;
      a = 0.5 * (alpha + beta);
    } else if (fx2 * p < c2 * m) {
      alpha = a;
      if (beta < 0)
        a = 2*alpha;
      else
        a = 0.5 * (alpha + beta);
    } else
      break;
    iter--;
    if (iter == 0)
      throw std::runtime_error("Too many iterations in BisectionWeakWolfe line search");
  }
}
};
}
