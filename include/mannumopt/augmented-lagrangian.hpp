#pragma once

#include <vector>
#include <memory>

#include <mannumopt/fwd.hpp>

namespace mannumopt {

template<typename Scalar, int Dim>
struct AugmentedLagrangian : Algo<Scalar,Dim> {
  MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

  using Algo<Scalar,Dim>::fxtol2;
  using Algo<Scalar,Dim>::maxIter;

  template <typename CFunctor, typename EFunctor>
  struct AL {
    CFunctor& C;
    EFunctor& E;

    Scalar mu;
    VectorS lambda = VectorS::Zero();

    // Temp datas
    RowVectorS cx;
    VectorS e;
    MatrixS ex;

    AL (CFunctor& C, EFunctor& E, Scalar mu0 = 1000) : C(C), E(E), mu(mu0) {}

    void f_fx(const VectorS& X, double& f, RowVectorS& fx)
    {
      C.f_fx(X, c, cx);
      E.f_fx(X, e, ex);

      //f = c - lambda.dot(e) + mu * e.squaredNorm() / 2;
      f = c + (- lambda + 0.5 * mu * e).dot(e);
      fx.noalias() = cx + (- lambda.transpose() + mu * e.transpose()) * ex;
    }

    void update(const VectorS& X)
    {
      C.f(X, c);
      lamda.noalias() -= mu * c;
      mu *= 5;
    }
  };

  Scalar mu0 = 1000;

  MatrixS fxx;

  RowVectorS fx1, fx2;

  VectorS p, x2;

  template<typename CFunctor, typename EFunctor, typename InnerAlgo, typename InnerLineSearch>
  bool minimize(CFunctor& C, EFunctor& E, VectorS& x1, InnerAlgo& ialgo = InnerAlgo(), InnerLineSearch ils = InnerLineSearch())
  {
    size_type iter = 0;

    AL<CFunctor, EFunctor> al (C, E, mu0);

    Scalar f1, f2;
    func.f_fx(x1, f2, fx2);
    VectorS x2;

    while(true) {
      x2 = x1;
      bool success = ialgo.minimize(al, x2, ils);
      if (success) {
        E.f(x2, al.e);
        if (al.e.squaredNorm() < etol2) {
        }
        x1 = x2;
        lamda.noalias() -= mu * c;
        mu *= 5;
      } else
        al.mu = 5 * al.mu;

      f1 = f2;
      fx1.swap(fx2);

      // Check termination criterion
      if (fx1.squaredNorm() < fxtol2)
        return true;
      if (iter > maxIter)
        return false;

      p = - fxx * fx1.transpose();
      Scalar a = 1.;
      ls(func, x1, p, f1, fx1, a, x2);

      x2.swap(x1);

      func.f_fx(x1, f2, fx2);
      VectorS y = fx2 - fx1;
      // s = a * p
      // rho = 1 / (y^T s)
      // rho_a = rho * a = 1 / (y^T p)
      Scalar rho_a = 1 / y.dot(p);

      //      H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
      // i.e. H = (I - rho_a p y^T) H (I - rho_a y p^T) + rho_a a p p^T
      auto I = MatrixS::Identity();
      fxx = (I - rho_a * p * y.transpose()) * fxx * (I - rho_a * y * p.transpose());
      fxx.noalias() += rho_a * a * p * p.transpose();

      ++iter;
    }
  }

};

}
