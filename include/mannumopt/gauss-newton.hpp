#pragma once

#include <mannumopt/fwd.hpp>
#include <mannumopt/function.hpp>

#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace mannumopt {

namespace internal {
  template<typename Scalar, int XDim, int TDim>
  struct traits<GaussNewton<Scalar, XDim, TDim>> {
    static constexpr bool for_vector_function = true;
  };
} // namespace internal

template<typename Scalar, int XDim, int TDim>
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

  GaussNewton(int xdim, int tdim) :
    exx(tdim,tdim),
    ex(tdim),
    p(tdim),
    x2(xdim)
  {}

  GaussNewton(int dim) : GaussNewton(dim, dim)
  {
    static_assert(XDim == TDim, "Dimensions must be equals");
  }

  GaussNewton() : GaussNewton(XDim, TDim)
  {
    static_assert(XDim != Eigen::Dynamic && TDim != Eigen::Dynamic, "You must provide dimensions");
  }

  template<typename LineSearch, typename VectorValuedFunctor, typename IntegrateFunctor, class Decomposition = Eigen::LDLT<MatrixTT> >
  bool minimize(VectorValuedFunctor& func, IntegrateFunctor integrate, VectorX& x1, LineSearch& ls, Decomposition dec = Decomposition())
  {
    iter = 0;

    typedef typename VectorValuedFunctor::VectorN ValueType;
    typedef typename VectorValuedFunctor::MatrixNT DerivativeType;

    auto n (func.dimension());

    ValueType f1(n);
    DerivativeType fx1(n, exx.rows());

    struct Norm2 : Function<Scalar, XDim, TDim> {
      void f(const VectorX& X, Scalar& fn) override
      {
        func.residual(X, fn);
      }
      void f_fx(const VectorX& X, Scalar& fn, RowVectorT& fx) override
      {
        func.f_fx(X, f_, fx_);
        fn = .5 * f_.squaredNorm();
        fx.noalias() = f_.transpose() * fx_;
      }

      VectorValuedFunctor& func;
      ValueType& f_;
      DerivativeType& fx_;

      Norm2 (VectorValuedFunctor& func, ValueType& f, DerivativeType& fx)
        :func(func), f_(f), fx_(fx) {}
    } funcLs{func, f1, fx1};

    while(true) {
      // e(x) = .5 * ||f(x)||^2
      // de/dx = f(x)^T df/dx(x)
      // d2e/dx2 ~ df/dx(x)^T df/dx(x)
      func.f_fx(x1, f1, fx1);
      double e = .5*f1.squaredNorm();
      ex.noalias() = f1.transpose() * fx1;

      // Check termination criterion: ||de/dx||^2 < fxtol2
      if (e < xtol) {
        this->print(true, "last_iter", iter, "*cost*", e,
            "grad", ex.norm());
        return true;
      }
      if (ex.squaredNorm() < fxtol2) {
        this->print(true, "last_iter", iter, "cost", e,
            "*grad*", ex.norm());
        return true;
      }
      if (iter > maxIter) {
        this->print(true, "*last_iter*", iter, "cost", e,
            "*grad*", ex.norm());
        return false;
      }

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

  template<typename LineSearch, typename VectorValuedFunctor, typename IntegrateFunctor, class Decomposition = Eigen::LDLT<MatrixTT> >
  bool minimize(VectorValuedFunctor& func, IntegrateFunctor integrate, VectorX& x1)
  {
    LineSearch ls;
    return minimize<LineSearch, VectorValuedFunctor, IntegrateFunctor, Decomposition>(func, integrate, x1, ls);
  }

  template<typename LineSearch, typename VectorValuedFunctor, class Decomposition = Eigen::LDLT<MatrixTT> >
  bool minimize(VectorValuedFunctor& func, VectorX& x, LineSearch& ls)
  {
    static_assert(XDim == TDim, "Variable space and tangent space must have the same dimension");
    auto integrate = &internal::vector_space_addition<Scalar, XDim>;
    return minimize<LineSearch, VectorValuedFunctor, decltype(integrate), Decomposition>(func, integrate, x, ls);
  }

  template<typename LineSearch, typename VectorValuedFunctor, class Decomposition = Eigen::LDLT<MatrixTT> >
  bool minimize(VectorValuedFunctor& func, VectorX& x)
  {
    LineSearch ls;
    return minimize<LineSearch, VectorValuedFunctor, Decomposition>(func, x, ls);
  }
};

}
