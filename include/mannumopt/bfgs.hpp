#pragma once

#include <iostream>
#include <iomanip>

#include <mannumopt/fwd.hpp>

namespace mannumopt {

template<typename Scalar, int Dim>
struct BFGS : Algo<Scalar,Dim> {
  MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

  using Algo<Scalar,Dim>::fxtol2;
  using Algo<Scalar,Dim>::maxIter;
  using Algo<Scalar,Dim>::iter;

  MatrixS fxx_i;

  RowVectorS fx1, fx2;

  VectorS p, x2;

  template<typename Functor, typename IntegrateFunctor, typename LineSearch>
  bool minimize(Functor& func, IntegrateFunctor integrate, VectorS& x1, LineSearch ls = LineSearch())
  {
    iter = 0;

    Scalar f1, f2;
    func.f_fx(x1, f2, fx2);

    while(true) {
      f1 = f2;
      fx1.swap(fx2);

      // Check termination criterion
      if (fx1.squaredNorm() < fxtol2)
        return true;
      if (iter > maxIter)
        return false;

      p = - fxx_i * fx1.transpose();
      Scalar a = 1.;
      ls(func, integrate, x1, p, f1, fx1, a, x2);

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
      MatrixS C (I - rho_a *p * y.transpose());
      fxx_i = (C * fxx_i * C.transpose()).eval();
      fxx_i.noalias() += rho_a * a * p * p.transpose();

      this->print(iter%10==0, "iter", iter, "cost", f1, "grad", fx1.norm(), "step", a);

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

