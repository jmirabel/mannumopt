#pragma once

#include <iostream>
#include <iomanip>

#include <mannumopt/fwd.hpp>
#include <mannumopt/decomposition/choleski.hpp>

namespace mannumopt {

template<typename Scalar, int Dim>
struct NewtonTR : Algo<Scalar,Dim> {
  MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

  using Algo<Scalar,Dim>::fxtol2;
  using Algo<Scalar,Dim>::maxIter;
  using Algo<Scalar,Dim>::iter;

  MatrixS fxx;

  RowVectorS fx;

  VectorS p, x2;

  Scalar eta = 0.1;
  Scalar u_maxstep = 2.;

  template<typename Functor, typename TrustRegion>
  bool minimize(Functor& func, VectorS& x, TrustRegion tr = TrustRegion())
  {
    iter = 0;
    Scalar maxstep = u_maxstep / 2;

    Scalar f, fn;
    VectorS xn;

    while(true) {
      func.f_fx_fxx(x, f, fx, fxx);

      // Check termination criterion
      if (fx.squaredNorm() < fxtol2)
        return true;
      if (iter > maxIter)
        return false;

      Scalar p_norm;
      tr(fx, fxx, maxstep, p, p_norm);

      func.x_add(xn, x, p);
      func.f(xn, fn);

      Scalar rho = (fn - f) / (fx.dot(p) + 0.5 * p.dot(fxx * p));
      if (rho > eta) x.swap(xn);

      if (this->verbose)
        print(iter, f, fx.norm(), maxstep, p_norm, rho);

      if (rho < 0.25)
        maxstep *= 0.25;
      else if (rho > 0.75 && p_norm == maxstep)
        maxstep = std::min(2*maxstep, u_maxstep);

      ++iter;
    }
  }

  void print(size_type iter, Scalar cost, Scalar grad, Scalar maxstep, Scalar step, Scalar rho)
  {
    if (iter % 10 == 0)
      std::cout << "iter \t cost \t grad \t maxstep \t step \t rho\n";

    std::cout << std::setw(4) << iter << "  ";
    std::cout << std::scientific << std::setprecision(5) << cost << "  ";
    std::cout << grad << "  ";
    std::cout << std::fixed << std::setprecision(4)
      << maxstep << "  " << step << "  " << rho << '\n';
  }
};

template<typename Scalar, int Dim>
struct NewtonLS : Algo<Scalar,Dim> {
  MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

  using Algo<Scalar,Dim>::fxtol2;
  using Algo<Scalar,Dim>::maxIter;
  using Algo<Scalar,Dim>::iter;

  MatrixS fxx;

  RowVectorS fx;

  VectorS p, x2;

  Scalar eta = 0.1;
  Scalar u_maxstep = 2.;

  template<typename Functor, typename LineSearch>
  bool minimize(Functor& func, VectorS& x, LineSearch ls = LineSearch())
  {
    iter = 0;
    Scalar maxstep = u_maxstep / 2;

    Scalar f;
    VectorS xn;

    ApproxLDLT<MatrixS> ldlt;

    while(true) {
      func.f_fx_fxx(x, f, fx, fxx);

      // Check termination criterion
      if (fx.squaredNorm() < fxtol2)
        return true;
      if (iter > maxIter)
        return false;

      p = ldlt.compute(fxx).solve(- fx.transpose());
      Scalar a = 1.;
      ls(func, x, p, f, fx, a, xn);

      xn.swap(x);

      if (this->verbose)
        print(iter, f, fx.norm(), a);

      ++iter;
    }
  }

  void print(size_type iter, Scalar cost, Scalar grad, Scalar step)
  {
    if (iter % 10 == 0)
      std::cout << "iter \t cost \t      grad \t step\n";

    std::cout << std::setw(4) << iter << "  ";
    std::cout << std::scientific << std::setprecision(5) << cost << "  ";
    std::cout << grad << "  ";
    std::cout << std::fixed << std::setprecision(4) << step << '\n';
  }
};

}
