#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt::lineSearch {
template<typename Scalar, int Dim>
struct Armijo {
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

Scalar r = 0.5;
Scalar c = 0.1;
Scalar amin = 1e-8;

template<typename Functor>
void operator() (Functor& func,
    const VectorS& x, const VectorS& p, Scalar f, const RowVectorS& fx,
    Scalar& a, VectorS& x2)
{
  Scalar m = fx * p;
  if (m >= 0)
    throw std::runtime_error("Not a valid descent direction");
  Scalar f2;
  while(true) {
    func.x_add(x2, x, a*p);
    func.f(x2, f2);
    if (f2 < f + c * a * m) break;
    a *= r;
    if (a < amin)
      throw std::runtime_error("Not a valid descent direction");
  }
}
};

/// The algorithm comes from
/// https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf
template<typename Scalar, int Dim>
struct BisectionWeakWolfe {
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

Scalar r = 0.5;
Scalar c1 = 0.1;
Scalar c2 = 0.9;
Scalar amin = 1e-8;

template<typename Functor>
void operator() (Functor& func,
    const VectorS& x, const VectorS& p, Scalar f, const RowVectorS& fx,
    Scalar& a, VectorS& x2)
{
  Scalar m = fx * p;
  if (m >= 0)
    throw std::runtime_error("Not a valid descent direction");

  Scalar f2;
  RowVectorS fx2;
  Scalar alpha = 0.;
  Scalar beta = -1.;

  a = 1.;
  while(true) {
    func.x_add(x2, x, a*p);
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
  }
}
};
}
