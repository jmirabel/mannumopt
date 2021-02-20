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

  template<typename VectorValuedFunctor, typename IntegrateFunctor, typename TrustRegion>
  bool minimize(VectorValuedFunctor& func, IntegrateFunctor integrate, VectorS& x, TrustRegion tr = TrustRegion())
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

      integrate(xn, x, p);
      func.f(xn, fn);

      Scalar rho = (fn - f) / (fx.dot(p) + 0.5 * p.dot(fxx * p));
      if (rho > eta) x.swap(xn);

      this->print(iter % 10 == 0,
          "iter", iter, "cost", f, "grad", fx.norm(),
          "maxstep", maxstep, "step", p_norm, "rho", rho);

      if (rho < 0.25)
        maxstep *= 0.25;
      else if (rho > 0.75 && p_norm == maxstep)
        maxstep = std::min(2*maxstep, u_maxstep);

      ++iter;
    }
  }

  template<typename Functor, typename TrustRegion>
  bool minimize(Functor& func, VectorS& x, TrustRegion tr = TrustRegion())
  {
    return minimize(func, &internal::vector_space_addition<Scalar, Dim>, x, tr); 
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

  template<typename VectorValuedFunctor, typename IntegrateFunctor, typename LineSearch, class Decomposition = Eigen::LDLT<Eigen::Matrix<Scalar, Dim, Dim>> >
  bool minimize(VectorValuedFunctor& func, IntegrateFunctor integrate, VectorS& x, LineSearch ls = LineSearch())
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
      ls(func, integrate, x, p, f, fx, a, xn);

      xn.swap(x);

      this->print(iter % 10 == 0,
          "iter", iter, "cost", f, "grad", fx.norm(), "step", a);

      ++iter;
    }
  }

  template<typename Functor, typename LineSearch>
  bool minimize(Functor& func, VectorS& x, LineSearch ls = LineSearch())
  {
    return minimize(func, &internal::vector_space_addition<Scalar, Dim>, x, ls); 
  }
};

}
