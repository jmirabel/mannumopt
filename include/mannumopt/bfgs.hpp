#pragma once

#include <iostream>
#include <iomanip>

#include <mannumopt/fwd.hpp>

namespace mannumopt {

namespace internal {
  template<typename Scalar, int XDim, int TDim>
  struct traits<BFGS<Scalar, XDim, TDim>> {
    constexpr static bool for_vector_function = false;
  };
} // namespace internal

template<typename Scalar, int XDim, int TDim>
struct BFGS : Algo<Scalar,XDim,TDim> {
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  using AlgoBase::fxtol2;
  using AlgoBase::maxIter;
  using AlgoBase::iter;

  MatrixTT fxx_i, C;

  RowVectorT fx1, fx2;
  VectorT p;
  VectorX x2;

  BFGS(int xdim, int tdim) :
    fxx_i(MatrixTT::Identity(tdim, tdim)),
    C(tdim, tdim),
    fx1(tdim),
    fx2(tdim),
    p(tdim),
    x2(xdim)
  {}

  BFGS(int dim) : BFGS(dim, dim)
  {
    static_assert(XDim == TDim, "Dimensions must be equals");
  }

  BFGS() : BFGS(XDim, TDim)
  {
    static_assert(XDim != Eigen::Dynamic && TDim != Eigen::Dynamic, "You must provide dimensions");
  }

  template<typename LineSearch, typename Functor, typename IntegrateFunctor>
  bool minimize(Functor& func, IntegrateFunctor integrate, VectorX& x1, LineSearch& ls)
  {
    iter = 0;

    Scalar f1, f2;
    func.f_fx(x1, f2, fx2);

    while(true) {
      f1 = f2;
      fx1.swap(fx2);

      // Check termination criterion
      if (fx1.squaredNorm() < fxtol2) {
        this->print(true, "last_iter", iter, "cost", f1, "*grad*", fx1.norm());
        return true;
      }
      if (iter > maxIter) {
        this->print(true, "*last_iter*", iter, "cost", f1, "grad", fx1.norm());
        return false;
      }

      p = - fxx_i * fx1.transpose();
      Scalar a = 1.;
      ls(func, integrate, x1, p, f1, fx1, a, x2);

      x2.swap(x1);

      func.f_fx(x1, f2, fx2);
      VectorT y = fx2 - fx1;
      // s = a * p
      // rho = 1 / (y^T s)
      // rho_a = rho * a = 1 / (y^T p)
      Scalar rho_a = 1 / y.dot(p);

      //      H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
      // i.e. H = (I - rho_a p y^T) H (I - rho_a y p^T) + rho_a a p p^T
      auto I = MatrixTT::Identity(fxx_i.rows(), fxx_i.cols());
      C.noalias() = (I - rho_a *p * y.transpose());
      fxx_i = (C * fxx_i * C.transpose()).eval();
      fxx_i.noalias() += rho_a * a * p * p.transpose();

      this->print(iter%10==0, "iter", iter, "cost", f1, "grad", fx1.norm(), "step", a);

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

} // namespace mannumopt
