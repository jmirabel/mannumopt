#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt {

template<typename Scalar, int XDim, int TDim, int ECDim>
struct AugmentedLagrangian : Algo<Scalar,XDim,TDim> {
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  typedef Eigen::Matrix<Scalar, ECDim, 1> VectorE;
  typedef Eigen::Matrix<Scalar, ECDim, TDim> MatrixET;

  Scalar etol2;
  using AlgoBase::fxtol2;
  using AlgoBase::maxIter;
  using AlgoBase::iter;

  template <typename CFunctor, typename EFunctor>
  struct AL {
    CFunctor& C;
    EFunctor& E;

    const VectorE& lambda;

    const Scalar& mu;

    // Temp datas
    double c;
    RowVectorT cx;
    VectorE e;
    MatrixET ex;

    AL (CFunctor& C, EFunctor& E, VectorE& lambda, Scalar& mu, auto tdim)
      : C(C), E(E), lambda(lambda), mu(mu),
      cx(tdim), e(lambda.size()), ex(lambda.size(), tdim)
    {}

    void f(const VectorX& X, double& f)
    {
      C.f(X, c);
      E.f(X, e);

      //f = c - lambda.dot(e) + mu * e.squaredNorm() / 2;
      f = c + (- lambda + 0.5 * mu * e).dot(e);
    }

    void f_fx(const VectorX& X, double& f, RowVectorT& fx)
    {
      C.f_fx(X, c, cx);
      E.f_fx(X, e, ex);

      //f = c - lambda.dot(e) + mu * e.squaredNorm() / 2;
      f = c + (- lambda + 0.5 * mu * e).dot(e);
      fx.noalias() = cx + (- lambda.transpose() + mu * e.transpose()) * ex;
    }
  };

  int tdim;

  Scalar mu = 1e-3;

  VectorE lambda;
  VectorX x2;

  AugmentedLagrangian(int xdim = XDim, int tdim = TDim, int ecdim = ECDim)
    : tdim(tdim), lambda(VectorE::Zero(ecdim)), x2(xdim) {}

  template<typename CFunctor, typename EFunctor, typename IntegrateFunctor, typename InnerAlgo, typename InnerLineSearch>
  bool minimize(CFunctor& C, EFunctor& E, IntegrateFunctor integrate, VectorX& x1, InnerAlgo& ialgo = InnerAlgo(), InnerLineSearch ils = InnerLineSearch())
  {
    iter = 0;

    AL<CFunctor, EFunctor> al {C, E, lambda, mu, x1.size()};

    VectorX x2 (x1);

    E.f(x1, al.e);
    Scalar feas = al.e.squaredNorm(), prevfeas = feas;

    while(mu < 1e10 && iter < maxIter) {
      // Compute tolerance for sub-problem.
      bool reached_desired_fxtol = (feas < 10*etol2);
      if (reached_desired_fxtol)
        ialgo.fxtol2 = fxtol2;
      else
        ialgo.fxtol2 = fxtol2 * std::sqrt(feas / (10*etol2));

      bool success = ialgo.minimize(al, integrate, x2, ils);
      if (success) {
        x1 = x2;

        E.f(x1, al.e);
        feas = al.e.squaredNorm();

        // Check the KKT condition:
        // optimality: cx - lambda^T * ex = 0 => ensured by the inner algorithm.
        // feasibility: e = 0
        if (reached_desired_fxtol && feas < etol2)
          return true;
        lambda.noalias() -= mu * al.e;
        if (feas > prevfeas / 2)
          mu *= 25;
        else
          mu *= 5;
        prevfeas = feas;
      } else {
        mu *= 5;
        x2 = x1;
      }

      if (this->verbose()) {
        C.f_fx(x1, al.c, al.cx);
        E.f_fx(x1, al.e, al.ex);
      }
      this->print(ialgo.verbose() || iter%10 == 0,
          "", (success ? ' ' : '!'),
          "iter", iter,
          "cost", al.c,
          "feas", al.e.squaredNorm(),
          "optim", (al.cx - lambda.transpose()*al.ex).squaredNorm(),
          "inner_its", ialgo.iter,
          "mu", mu);

      ++iter;
    }
    return false;
  }

  template<typename CFunctor, typename EFunctor, typename InnerAlgo, typename InnerLineSearch>
  bool minimize(CFunctor& C, EFunctor& E, VectorX& x1, InnerAlgo& ialgo = InnerAlgo(), InnerLineSearch ils = InnerLineSearch())
  {
    static_assert(XDim == TDim, "Variable space and tangent space must have the same dimension");
    return minimize(C, E, &internal::vector_space_addition<Scalar, XDim>, x1, ialgo, ils);
  }

};

}
