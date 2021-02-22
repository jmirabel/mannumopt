#pragma once

#include <mannumopt/fwd.hpp>

#include <Eigen/Eigenvalues>

namespace mannumopt::trustRegion {
template<typename Scalar, int Dim>
struct Cauchy {
MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim, S);

void operator() (const RowVectorS& fx, const MatrixS& fxx,
    Scalar delta, VectorS& p, Scalar& p_norm)
{
  Eigen::SelfAdjointEigenSolver<MatrixS> solver (fxx);
  bool positive_definite = (solver.eigenvalues().array() > 0).all();
  if (positive_definite) {
    auto V = solver.eigenvectors();
    p = - V * solver.eigenvalues().cwiseInverse().asDiagonal() * V.transpose() * fx.transpose();
    p_norm = p.norm();
    if (p_norm <= delta) {
      return;
    }
  }

  Scalar fx_norm (fx.norm());
  p = - fx * delta / fx.norm();
  p_norm = delta;

  Scalar fx_fxx_fx;
  if (positive_definite || (fx_fxx_fx = fx * fxx * fx.transpose()) > 0) {
    Scalar tau (fx_norm*fx_norm*fx_norm / delta / fx_fxx_fx);
    if (tau < 1.) {
      p *= tau;
      p_norm *= tau;
    }
  }
}
};

}
