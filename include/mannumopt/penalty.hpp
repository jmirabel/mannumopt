#pragma once

#include <mannumopt/fwd.hpp>
#include <mannumopt/function.hpp>

namespace mannumopt {

template<typename Scalar, int XDim, int TDim, int ECDim>
struct Penalty : Algo<Scalar,XDim,TDim> {
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  typedef Eigen::Matrix<Scalar, ECDim, 1> VectorE;
  typedef Eigen::Matrix<Scalar, ECDim, TDim> MatrixET;

  Scalar etol2;
  using AlgoBase::fxtol2;
  using AlgoBase::maxIter;
  using AlgoBase::iter;

  template <typename CFunctor, typename EFunctor>
  struct PenalizedCost : Function<Scalar, XDim, TDim> {
    CFunctor& C;
    EFunctor& E;

    const Scalar& mu;

    // Temp datas
    double c;
    RowVectorT cx;
    VectorE e;
    MatrixET ex;

    PenalizedCost (CFunctor& C, EFunctor& E, Scalar& mu, auto tdim)
      : C(C), E(E), mu(mu), cx(tdim)
    {
      int ne = E.dimension();
      e.resize(ne);
      ex.resize(ne, tdim);
    }

    void f(const VectorX& X, double& f) override
    {
      C.f(X, c);
      E.f(X, e);

      f = c + 0.5 * mu * e.squaredNorm();
    }

    void f_fx(const VectorX& X, double& f, RowVectorT& fx) override
    {
      C.f_fx(X, c, cx);
      E.f_fx(X, e, ex);

      f = c + 0.5 * mu * e.squaredNorm();
      fx.noalias() = cx + mu * e.transpose() * ex;
    }

    double cost(const VectorX& X)
    {
      C.f(X, c);
      return c;
    }
  };

  template <typename CFunctor, typename EFunctor, int NDim>
  struct PenalizedResidual : VectorFunction<Scalar, NDim, XDim, TDim> {
    typedef VectorFunction<Scalar, NDim, XDim, TDim> Base;
    using typename Base::VectorN;
    using typename Base::MatrixNT;

    CFunctor& C;
    EFunctor& E;

    const Scalar& mu;

    // Temp datas
    typename CFunctor::Output c;
    typename CFunctor::Derivative cx;
    typename EFunctor::Output e;
    typename EFunctor::Derivative ex;

    PenalizedResidual (CFunctor& C, EFunctor& E, Scalar& mu, auto tdim)
      : C(C), E(E), mu(mu)
    {
      int nc = C.dimension();
      c.resize(nc);
      cx.resize(nc, tdim);
      int ne = E.dimension();
      e.resize(ne);
      ex.resize(ne, tdim);
    }

    int dimension() override
    {
      return C.dimension() + E.dimension();
    }

    void f(const VectorX& X, VectorN& f) override
    {
      C.f(X, c);
      E.f(X, e);
      f.head(C.dimension()) = c;
      f.tail(E.dimension()) = std::sqrt(mu) * e;
    }

    void f_fx(const VectorX& X, VectorN& f, MatrixNT& fx) override
    {
      C.f_fx(X, c, cx);
      E.f_fx(X, e, ex);

      f.head(C.dimension()) = c;
      fx.topRows(C.dimension()) = cx;

      Scalar sqrt_mu = std::sqrt(mu);
      f.tail(E.dimension()) = sqrt_mu * e;
      fx.bottomRows(E.dimension()) = sqrt_mu * ex;
    }

    double cost(const VectorX& X)
    {
      C.f(X, c);
      return 0.5 * c.squaredNorm();
    }
  };

  template <typename CFunctor, typename EFunctor, typename InnerAlgo, bool for_vector_function = internal::traits<InnerAlgo>::for_vector_function> struct PenaltyFuncTpl;
  template <typename CFunctor, typename EFunctor, typename InnerAlgo> struct PenaltyFuncTpl<CFunctor, EFunctor, InnerAlgo, true> {
    static constexpr int Dim = (ECDim == Eigen::Dynamic || CFunctor::VectorN::ColsAtCompileTime == Eigen::Dynamic )? Eigen::Dynamic : ECDim + CFunctor::VectorN::ColsAtCompileTime;
    typedef PenalizedResidual<CFunctor, EFunctor, Dim> type;
  };
  template <typename CFunctor, typename EFunctor, typename InnerAlgo> struct PenaltyFuncTpl<CFunctor, EFunctor, InnerAlgo, false> {
    typedef PenalizedCost<CFunctor, EFunctor> type;
  };

  int tdim, ecdim;

  Scalar mu = 1e-3;

  Penalty(int xdim, int tdim, int ecdim)
    : tdim(tdim), ecdim(ecdim)
  { (void)xdim; }

  Penalty(int dim, int ecdim) : Penalty(dim, dim, ecdim)
  {
    static_assert(XDim == TDim, "Dimensions must be equals");
  }

  Penalty(int ecdim) : Penalty(XDim, TDim, ecdim)
  {
    static_assert(XDim != Eigen::Dynamic && TDim != Eigen::Dynamic, "You must provide dimensions");
  }

  Penalty() : Penalty(XDim, TDim, ECDim)
  {
    static_assert(XDim != Eigen::Dynamic && TDim != Eigen::Dynamic && ECDim != Eigen::Dynamic, "You must provide dimensions");
  }

  template<typename InnerLineSearch, typename InnerAlgo, typename CFunctor, typename EFunctor, typename IntegrateFunctor>
  bool minimize(CFunctor& C, EFunctor& E, IntegrateFunctor integrate, VectorX& x1, InnerAlgo& ialgo, InnerLineSearch& ils)
  {
    iter = 0;

    typedef typename PenaltyFuncTpl<CFunctor, EFunctor, InnerAlgo>::type PenaltyFunc;

    PenaltyFunc pen {C, E, mu, tdim};

    Scalar f_mu_small = 5,
           f_mu_big   = 25;

    VectorX x2 (x1);

    E.f(x1, pen.e);
    Scalar feas = pen.e.squaredNorm(), prevfeas = feas;

    while(true) {
      // Compute tolerance for sub-problem.
      bool reached_desired_fxtol = (feas < 10*etol2);
      if (reached_desired_fxtol)
        ialgo.fxtol2 = fxtol2;
      else
        ialgo.fxtol2 = fxtol2 * std::sqrt(feas / (10*etol2));

      bool success = ialgo.minimize(pen, integrate, x2, ils);
      if (success) {
        x1 = x2;

        E.f(x1, pen.e);
        feas = pen.e.squaredNorm();

        // Check the KKT condition:
        // optimality: cx - lambda^T * ex = 0 => ensured by the inner algorithm.
        // feasibility: e = 0
        if (reached_desired_fxtol && feas < etol2) {
          this->print(true,
              1, "", ' ',
              "last_iter", iter,
              "cost", pen.cost(x1),
              "*feas*", feas,
              "inner_its", ialgo.iter);
          return true;
        }
        if (feas > prevfeas / 2)
          mu *= f_mu_big;
        else
          mu *= f_mu_small;
        prevfeas = feas;
      } else {
        mu *= f_mu_small;
        x2 = x1;
      }

      if (mu > 1e10 || iter > maxIter) {
        this->print(true,
            1, "", (success ? ' ' : '!'),
            (iter>maxIter ? "*last_iter*" : "last_iter"), iter,
            "cost", pen.cost(x1),
            "feas", feas,
            "inner_its", ialgo.iter,
            (mu > 1e10 ? "*mu*" : "mu"), mu);
        return false;
      }
      this->print(ialgo.verbose() || iter%10 == 0,
          1, "", (success ? ' ' : '!'),
          "iter", iter,
          "cost", pen.cost(x1),
          "feas", feas,
          "inner_its", ialgo.iter,
          "mu", mu);

      ++iter;
    }
    return false;
  }

  template<typename InnerLineSearch, typename InnerAlgo, typename CFunctor, typename EFunctor, typename IntegrateFunctor>
  bool minimize(CFunctor& C, EFunctor& E, IntegrateFunctor integrate, VectorX& x1, InnerAlgo& ialgo)
  {
    InnerLineSearch ls;
    return minimize(C, E, integrate, x1, ialgo, ls);
  }

  template<typename InnerLineSearch, typename InnerAlgo, typename CFunctor, typename EFunctor>
  bool minimize(CFunctor& C, EFunctor& E, VectorX& x1, InnerAlgo& ialgo, InnerLineSearch& ils)
  {
    static_assert(XDim == TDim, "Variable space and tangent space must have the same dimension");
    return minimize(C, E, &internal::vector_space_addition<Scalar, XDim>, x1, ialgo, ils);
  }

  template<typename InnerLineSearch, typename InnerAlgo, typename CFunctor, typename EFunctor>
  bool minimize(CFunctor& C, EFunctor& E, VectorX& x1, InnerAlgo& ialgo)
  {
    InnerLineSearch ls;
    return minimize(C, E, x1, ialgo, ls);
  }

};

}
