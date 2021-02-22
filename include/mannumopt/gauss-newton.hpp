#pragma once

#include <iostream>
#include <iomanip>

#include <mannumopt/fwd.hpp>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace mannumopt {

template<typename Scalar, int XDim, int TDim = XDim>
struct GaussNewton : Algo<Scalar,XDim,TDim> {
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  using AlgoBase::xtol;
  using AlgoBase::fxtol2;
  using AlgoBase::maxIter;
  using AlgoBase::iter;

  MatrixTT exx;

  RowVectorT ex;

  VectorT p;
  VectorX x2;

  GaussNewton(int xdim = XDim, int tdim = TDim) :
    exx(tdim,tdim),
    ex(tdim),
    p(tdim),
    x2(xdim)
  {}

  template<typename VectorValuedFunctor, typename IntegrateFunctor, typename LineSearch, class Decomposition = Eigen::LDLT<MatrixTT> >
  bool minimize(VectorValuedFunctor& func, IntegrateFunctor integrate, VectorX& x1, LineSearch ls = LineSearch())
  {
    iter = 0;

    typedef typename VectorValuedFunctor::ValueType ValueType;
    typedef typename VectorValuedFunctor::DerivativeType DerivativeType;

    auto n (func.dimension());

    ValueType f1(n);
    DerivativeType fx1(n, exx.rows());

    Decomposition dec (exx.size());

    struct Norm2 {
      void f(const VectorX& X, Scalar& fn) {
        func.f(X, f_);
        fn = .5 * f_.squaredNorm();
      }
      void f_fx(const VectorX& X, Scalar& fn, RowVectorT& fx)
      {
        func.f_fx(X, f_, fx_);
        fn = .5 * f_.squaredNorm();
        fx.noalias() = f_.transpose() * fx_;
      }

      VectorValuedFunctor& func;
      ValueType& f_;
      DerivativeType& fx_;
    } funcLs{func, f1, fx1};

    while(true) {
      // e(x) = .5 * ||f(x)||^2
      // de/dx = f(x)^T df/dx(x)
      // d2e/dx2 ~ df/dx(x)^T df/dx(x)
      func.f_fx(x1, f1, fx1);
      double e = .5*f1.squaredNorm();
      ex.noalias() = f1.transpose() * fx1;

      // Check termination criterion: ||de/dx||^2 < fxtol2
      if (e < xtol)
        return true;
      if (ex.squaredNorm() < fxtol2)
        return true;
      if (iter > maxIter)
        return false;

      // df/dx v = - f
      // or d2e/dx2 p = - de/dx^T

      // Solve d2e/dx2 p = - de/dx^T
      exx.noalias() = fx1.transpose() * fx1;
      dec.compute(exx);
      p = dec.solve(- ex.transpose());
      Scalar a = 1.;
      ls(funcLs, integrate, x1, p, e, ex, a, x2);

      x2.swap(x1);

      this->print(iter%10==0, "iter", iter, "cost", e,
          "grad", ex.norm(), "step", a);

      ++iter;
    }
  }

  template<typename VectorValuedFunctor, typename LineSearch, class Decomposition = Eigen::LDLT<MatrixTT> >
  bool minimize(VectorValuedFunctor& func, VectorX& x, LineSearch ls = LineSearch())
  {
    static_assert(XDim == TDim, "Variable space and tangent space must have the same dimension");
    auto integrate = &internal::vector_space_addition<Scalar, XDim>;
    return minimize<VectorValuedFunctor, decltype(integrate), LineSearch, Decomposition>(func, integrate, x, ls); 
  }
};

}
