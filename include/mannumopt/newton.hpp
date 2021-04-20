#pragma once

#include <iostream>
#include <iomanip>

#include <mannumopt/fwd.hpp>
#include <mannumopt/decomposition/choleski.hpp>

namespace mannumopt {

namespace internal {
  template<typename Scalar, int XDim, int TDim>
  struct traits<NewtonTR<Scalar, XDim, TDim>> {
    static constexpr bool for_vector_function = false;
  };
  template<typename Scalar, int XDim, int TDim>
  struct traits<NewtonLS<Scalar, XDim, TDim>> {
    static constexpr bool for_vector_function = false;
  };
} // namespace internal

template<typename Scalar, int XDim, int TDim>
struct NewtonTR : Algo<Scalar,XDim,TDim> {
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  using AlgoBase::fxtol2;
  using AlgoBase::maxIter;
  using AlgoBase::iter;

  MatrixTT fxx;

  RowVectorT fx;
  VectorT p;
  VectorX xn;

  Scalar eta = 0.1;
  Scalar u_maxstep = 2.;

  NewtonTR(int xdim, int tdim) :
    fxx(tdim,tdim),
    fx(tdim),
    p(tdim),
    xn(xdim)
  {}

  NewtonTR(int dim) : NewtonTR(dim, dim)
  {
    static_assert(XDim == TDim, "Dimensions must be equals");
  }

  NewtonTR() : NewtonTR(XDim, TDim)
  {
    static_assert(XDim != Eigen::Dynamic && TDim != Eigen::Dynamic, "You must provide dimensions");
  }

  template<typename TrustRegion, typename VectorFunctor, typename IntegrateFunctor>
  bool minimize(VectorFunctor& func, IntegrateFunctor integrate, VectorX& x, TrustRegion& tr)
  {
    iter = 0;
    Scalar maxstep = u_maxstep / 2;

    Scalar f, fn;

    while(true) {
      func.f_fx_fxx(x, f, fx, fxx);

      // Check termination criterion
      if (fx.squaredNorm() < fxtol2) {
        this->print(true, "last_iter", iter, "cost", f, "*grad*", fx.norm());
        return true;
      }
      if (iter > maxIter) {
        this->print(true, "*last_iter*", iter, "cost", f, "grad", fx.norm());
        return false;
      }

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

  template<typename TrustRegion, typename Functor, typename IntegrateFunctor>
  bool minimize(Functor& func, IntegrateFunctor integrate, VectorX& x)
  {
    TrustRegion ls;
    return minimize(func, integrate, x, ls); 
  }

  template<typename TrustRegion, typename Functor>
  bool minimize(Functor& func, VectorX& x, TrustRegion& ls)
  {
    static_assert(XDim == TDim, "Variable space and tangent space must have the same dimension");
    return minimize(func, &internal::vector_space_addition<Scalar, XDim>, x, ls); 
  }

  template<typename TrustRegion, typename Functor>
  bool minimize(Functor& func, VectorX& x)
  {
    TrustRegion ls;
    return minimize(func, x, ls); 
  }
};

template<typename Scalar, int XDim, int TDim>
struct NewtonLS : Algo<Scalar,XDim,TDim> {
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  using AlgoBase::fxtol2;
  using AlgoBase::maxIter;
  using AlgoBase::iter;

  MatrixTT fxx;

  RowVectorT fx;

  VectorT p;
  VectorX xn;

  NewtonLS(int xdim, int tdim) :
    fxx(tdim,tdim),
    fx(tdim),
    p(tdim),
    xn(xdim)
  {}

  NewtonLS(int dim) : NewtonLS(dim, dim)
  {
    static_assert(XDim == TDim, "Dimensions must be equals");
  }

  NewtonLS() : NewtonLS(XDim, TDim)
  {
    static_assert(XDim != Eigen::Dynamic && TDim != Eigen::Dynamic, "You must provide dimensions");
  }

  template<typename LineSearch, typename VectorFunctor, typename IntegrateFunctor, class Decomposition = ApproxLDLT<MatrixTT> >
  bool minimize(VectorFunctor& func, IntegrateFunctor integrate, VectorX& x, LineSearch& ls)
  {
    iter = 0;

    Scalar f;

    Decomposition dec(fxx.rows());

    while(true) {
      func.f_fx_fxx(x, f, fx, fxx);

      // Check termination criterion
      if (fx.squaredNorm() < fxtol2) {
        this->print(true, "last_iter", iter, "cost", f, "*grad*", fx.norm());
        return true;
      }
      if (iter > maxIter) {
        this->print(true, "*last_iter*", iter, "cost", f, "grad", fx.norm());
        return false;
      }

      p = dec.compute(fxx).solve(- fx.transpose());
      Scalar a = 1.;
      ls(func, integrate, x, p, f, fx, a, xn);

      xn.swap(x);

      this->print(iter % 10 == 0,
          "iter", iter, "cost", f, "grad", fx.norm(), "step", a);

      ++iter;
    }
  }

  template<typename LineSearch, typename Functor, typename IntegrateFunctor>
  bool minimize(Functor& func, IntegrateFunctor integrate, VectorX& x)
  {
    LineSearch ls;
    return minimize(func, integrate, x, ls); 
  }

  template<typename LineSearch, typename Functor>
  bool minimize(Functor& func, VectorX& x, LineSearch& ls)
  {
    static_assert(XDim == TDim, "Variable space and tangent space must have the same dimension");
    return minimize(func, &internal::vector_space_addition<Scalar, XDim>, x, ls); 
  }

  template<typename LineSearch, typename Functor>
  bool minimize(Functor& func, VectorX& x)
  {
    LineSearch ls;
    return minimize(func, x, ls); 
  }
};

}
