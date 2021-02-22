#pragma once

#include <iostream>

#include <mannumopt/fwd.hpp>
#include <mannumopt/decomposition/choleski.hpp>

namespace mannumopt::trustRegion {

template<typename Scalar>
void find_root(
    auto f,
    auto fu,
    Scalar thr,
    Scalar& u)
{
  Scalar v;
  while (fabs(v = f(u)) > thr) {
    u -= v / fu(u);
  }
  assert(u >= 0);
}

template<typename Scalar>
void find_root_of_polynom(
    Scalar a0,
    Scalar a1,
    Scalar a2,
    Scalar a3,
    Scalar a4,
    Scalar thr,
    Scalar& u)
{
  auto f = [&a0, &a1, &a2, &a3, &a4](Scalar u) -> Scalar {
    return (((a4*u + a3) * u + a2) * u + a1) * u + a0;
  };
  auto fu = [&a1, &a2, &a3, &a4](Scalar u) -> Scalar {
    return ((4*a4*u + 3*a3) * u + 2*a2) * u + a1;
  };

  assert(a4 < 0);

  find_root(f, fu, thr, u);
}

template<typename Scalar, int Dim>
struct TwoDimensional {
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim, S);

ApproxLDLT<MatrixS> ldlt;

void operator() (const RowVectorS& fx, const MatrixS& fxx,
    Scalar delta, VectorS& p, Scalar& p_norm)
{
  // sg = - fx * ||fx||^2 / (fx^T H fx)
  // sg = - fxx^-1 fx

  VectorS sn = - ldlt.compute(fxx).solve(fx.transpose());
  p_norm = sn.norm();
  if (p_norm <= delta) {
    p = sn;
    return;
  }
  VectorS sg = - fx * (fx.squaredNorm() / (fx.dot(fxx * fx.transpose())));
  //VectorS sg = - fx;

  Scalar a (sg.transpose() * fxx * sg),
         b (sn.transpose() * fxx * sn),
         c (sg.transpose() * fxx * sn),
         d (sg.squaredNorm()),
         e (sn.squaredNorm()),
         f (sg.dot(sn)),
         g0 (fx.dot(sg)),
         g1 (fx.dot(sn));

  if (d / f - 1 > - 1e-8) {
    p = sn * delta / p_norm;
    p_norm = delta;
    return;
  }

  Scalar u = 0;
  {
    Scalar g = b*b*g0*g0 + c*c*g0*g0 - 2*a*c*g0*g1 - 2*b*c*g0*g1 + a*a*g1*g1 + c*c*g1*g1,
           h = 2*b*e*g0*g0 + 2*c*f*g0*g0 - 2*c*d*g0*g1 - 2*c*e*g0*g1 - 2*a*f*g0*g1 - 2*b*f*g0*g1 + 2*a*d*g1*g1 + 2*c*f*g1*g1,
           i = e*e*g0*g0 + f*f*g0*g0 - 2*d*f*g0*g1 - 2*e*f*g0*g1 + d*d*g1*g1 + f*f*g1*g1;
    Eigen::Matrix<Scalar, 2, 2> D;
    D << d, f, f, e;
    find_root(
        [&a,&b,&c,&d,&e,&f,&g,&h,&i](Scalar u) -> Scalar {
          Scalar num = (i*u + h)* u + g;
          Scalar den = ( (d*u+a)*(e*u+b) - (f*u+c)*(f*u+c) );
          den *= den;
          return num / den;
        },
        [&a,&b,&c,&d,&e,&f,&g,&h,&i](Scalar u) -> Scalar {
          Scalar num  = (i*u + h)* u + g;
          Scalar numd = 2*i*u + h;
          Scalar den = ( (d*u+a)*(e*u+b) - (f*u+c)*(f*u+c) );
          Scalar dend = 2*(e*(d*u+a) + d*(e*u+b) - 2*f*(f*u+c)) * den;
          den *= den;
          return numd / den - num*dend / (den*den);
        },
        1e-3,
        u);
  }

  Eigen::Matrix<Scalar, 2, 2> M;
  M(0,0) = a + d*u;
  M(1,1) = b + e*u;
  M(0,1) = M(1,0) = c + f*u;

  Eigen::Matrix<Scalar, 2, 1> s, g;
  g << g0, g1;
  s = - M.inverse() * g;
  p = sg*s(0) + sn*s(1);
  p_norm = p.norm();
}
};

}
