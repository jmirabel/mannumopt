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
  //std::cout << u << ": " << f(u) << '\n';
  while (fabs(v = f(u)) > thr) {
    u -= v / fu(u);
    std::cout << u << ": " << f(u) << ' ' << fu(u) << '\n';
  }
  //std::cout << u << ": " << f(u) << '\n';
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

  std::cout << a0 << '+' << a1 << "*x+" << a2 << "*x**2+" << a3 << "*x**3+" << a4 << "*x**4" << std::endl;

  assert(a4 < 0);

  find_root(f, fu, thr, u);
}

/*
template<typename Scalar>
void find_root_of_polynom2(
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

  bool first = true;
  Scalar v0, v1, vu = 1., p = 1.;
  std::cout << u << ": " << f(u) << '\n';
  while (fabs(v1 = f(u)) > thr) {
    if (!first)
      vu = (v1 - v0) / p;
    p = - v1 / vu;
    u += p;
    v0 = v1;
    first = false;
    std::cout << u << ": " << f(u) << '\n';
  }
}
*/

template<typename Scalar, int Dim>
struct TwoDimensional {
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim, S);

ApproxLDLT<MatrixS> ldlt;

void operator() (const RowVectorS& fx, const MatrixS& fxx,
    Scalar delta, VectorS& p, Scalar& p_norm)
{
  // sg = - fx * ||fx||^2 / (fx^T H fx)
  // sg = - fxx^-1 fx
  //std::cout << fxx << '\n';
  //std::cout << fx << '\n';

  VectorS sn = - ldlt.compute(fxx).solve(fx.transpose());
  p_norm = sn.norm();
  if (p_norm <= delta) {
    //std::cout << "N " << delta;
    p = sn;
    return;
  }
  VectorS sg = - fx * (fx.squaredNorm() / (fx.dot(fxx * fx.transpose())));
  //VectorS sg = - fx;
  //std::cout << sg.transpose() << '\n' << sn.transpose() << '\n';

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

  //std::cout << a  << ' ' << b  << ' ' << c  << ' ' << d  << ' ' << e  << ' ' << f  << ' ' << g0 << ' ' << g1 << '\n' ;

  Scalar u = 0;
#if 1
  {
    Scalar g = b*b*g0*g0 + c*c*g0*g0 - 2*a*c*g0*g1 - 2*b*c*g0*g1 + a*a*g1*g1 + c*c*g1*g1,
           h = 2*b*e*g0*g0 + 2*c*f*g0*g0 - 2*c*d*g0*g1 - 2*c*e*g0*g1 - 2*a*f*g0*g1 - 2*b*f*g0*g1 + 2*a*d*g1*g1 + 2*c*f*g1*g1,
           i = e*e*g0*g0 + f*f*g0*g0 - 2*d*f*g0*g1 - 2*e*f*g0*g1 + d*d*g1*g1 + f*f*g1*g1;
    Eigen::Matrix<Scalar, 2, 2> D;
    D << d, f, f, e;
    std::cout << D.eigenvalues() << std::endl;
    std::cout << "("<<i<<"*x + "<<h<<")* x + "<<g<<") / ( ("<<d<<"*x+"<<a<<")*("<<e<<"*x+"<<b<<") - ("<<f<<"*x+"<<c<<")**2 )**2\n";
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

#else
  //Scalar scale = std::pow(a*b*c*d*e*f*g0*g1, 1/8);
  Scalar scale;
  //*
  scale =
    std::pow(std::abs(a ), 1/8.) *
    std::pow(std::abs(b ), 1/8.) *
    std::pow(std::abs(c ), 1/8.) *
    std::pow(std::abs(d ), 1/8.) *
    std::pow(std::abs(e ), 1/8.) *
    std::pow(std::abs(f ), 1/8.) *
    std::pow(std::abs(g0), 1/8.) *
    std::pow(std::abs(g1), 1/8.);
  a /= scale;
  b /= scale;
  c /= scale;
  d /= scale;
  e /= scale;
  f /= scale;
  g0 /= scale;
  g1 /= scale;
  delta /= scale;
  //std::cout << scale << '\n';
  // */

  Scalar p0 = -a*a*b*b*delta + 2*a*b*c*c*delta - c*c*c*c*delta + b*b*g0*g0 + c*c*g0*g0 - 2*a*c*g0*g1 - 2*b*c*g0*g1 + a*a*g1*g1 + c*c*g1*g1,
         p1 = -2*a*b*b*d*delta + 2*b*c*c*d*delta - 2*a*a*b*delta*e + 2*a*c*c*delta*e + 4*a*b*c*delta*f - 4*c*c*c*delta*f + 2*b*e*g0*g0 + 2*c*f*g0*g0 - 2*c*d*g0*g1 - 2*c*e*g0*g1 - 2*a*f*g0*g1 - 2*b*f*g0*g1 + 2*a*d*g1*g1 + 2*c*f*g1*g1,
         p2 = -b*b*d*d*delta - 4*a*b*d*delta*e + 2*c*c*d*delta*e - a*a*delta*e*e + 4*b*c*d*delta*f + 4*a*c*delta*e*f + 2*a*b*delta*f*f - 6*c*c*delta*f*f + e*e*g0*g0 + f*f*g0*g0 - 2*d*f*g0*g1 - 2*e*f*g0*g1 + d*d*g1*g1 + f*f*g1*g1,
         p3 = -2*b*d*d*delta*e - 2*a*d*delta*e*e + 4*c*d*delta*e*f + 2*b*d*delta*f*f + 2*a*delta*e*f*f - 4*c*delta*f*f*f,
         p4 = -d*d*delta*e*e + 2*d*delta*e*f*f - delta*f*f*f*f;

  scale = std::pow(p4, 1/4.);

  // Find u such that p0 + p1 u + p2 u^2 + p3 u^3 + p4 u^4 = 0
  find_root_of_polynom(p0, p1, p2, p3, p4, 1e-3, u);
  //find_root_of_polynom(p0, p1/scale, p2/(scale*scale), p3/(scale*scale*scale), 1., 1e-3, u);
  //u *= scale;
#endif

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
