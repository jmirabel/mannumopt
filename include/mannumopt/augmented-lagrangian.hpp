#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt {

template<typename Scalar, int Dim>
struct AugmentedLagrangian : Algo<Scalar,Dim> {
  MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

  Scalar etol2;
  using Algo<Scalar,Dim>::fxtol2;
  using Algo<Scalar,Dim>::maxIter;
  using Algo<Scalar,Dim>::iter;

  template <typename CFunctor, typename EFunctor>
  struct AL {
    CFunctor& C;
    EFunctor& E;

    Scalar mu;
    typedef typename EFunctor::Vector VectorE;
    typedef typename EFunctor::Matrix MatrixE;

    VectorE lambda = VectorE::Zero();

    // Temp datas
    double c;
    RowVectorS cx;
    VectorE e;
    MatrixE ex;

    AL (CFunctor& C, EFunctor& E, Scalar mu0) : C(C), E(E), mu(mu0) {}

    void f(const VectorS& X, double& f)
    {
      C.f(X, c);
      E.f(X, e);

      //f = c - lambda.dot(e) + mu * e.squaredNorm() / 2;
      f = c + (- lambda + 0.5 * mu * e).dot(e);
    }

    void f_fx(const VectorS& X, double& f, RowVectorS& fx)
    {
      C.f_fx(X, c, cx);
      E.f_fx(X, e, ex);

      //f = c - lambda.dot(e) + mu * e.squaredNorm() / 2;
      f = c + (- lambda + 0.5 * mu * e).dot(e);
      fx.noalias() = cx + (- lambda.transpose() + mu * e.transpose()) * ex;
    }
  };

  Scalar mu0 = 1e-3;

  MatrixS fxx;

  RowVectorS fx1, fx2;

  VectorS p, x2;

  template<typename CFunctor, typename EFunctor, typename IntegrateFunctor, typename InnerAlgo, typename InnerLineSearch>
  bool minimize(CFunctor& C, EFunctor& E, IntegrateFunctor integrate, VectorS& x1, InnerAlgo& ialgo = InnerAlgo(), InnerLineSearch ils = InnerLineSearch())
  {
    iter = 0;

    AL<CFunctor, EFunctor> al (C, E, mu0);

    VectorS x2 (x1);

    E.f(x1, al.e);
    Scalar feas = al.e.squaredNorm(), prevfeas = feas;

    while(al.mu < 1e10 && iter < maxIter) {
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
        al.lambda.noalias() -= al.mu * al.e;
        if (feas > prevfeas / 2)
          al.mu *= 25;
        else
          al.mu *= 5;
        prevfeas = feas;
      } else {
        al.mu *= 5;
        x2 = x1;
        std::cout << "ialgo failed\n";
      }

      if (this->verbose()) {
        C.f_fx(x1, al.c, al.cx);
        E.f_fx(x1, al.e, al.ex);
      }
      this->print(ialgo.verbose() || iter%10 == 0,
          "iter", iter,
          "cost", al.c,
          "feas", al.e.squaredNorm(),
          "optim", (al.cx - al.lambda.transpose()*al.ex).squaredNorm(),
          "inner_its", ialgo.iter,
          "mu", al.mu);

      ++iter;
    }
    return false;
  }

  template<typename CFunctor, typename EFunctor, typename InnerAlgo, typename InnerLineSearch>
  bool minimize(CFunctor& C, EFunctor& E, VectorS& x1, InnerAlgo& ialgo = InnerAlgo(), InnerLineSearch ils = InnerLineSearch())
  {
    return minimize(C, E, &internal::vector_space_addition<Scalar, Dim>, x1, ialgo, ils);
  }

};

}
