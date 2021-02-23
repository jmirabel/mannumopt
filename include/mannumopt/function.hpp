#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt {
template<typename Scalar, int XDim, int TDim = XDim>
struct Function
{
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  virtual ~Function() = default;

  virtual void f(const VectorX& X, double& f) = 0;

  virtual void f_fx(const VectorX& X, double& f, RowVectorT& fx) = 0;

  virtual void f_fx_fxx(const VectorX& X, double& f, RowVectorT& fx, MatrixTT& fxx)
  {
    throw std::logic_error("mannumopt::Function::f_fx_fxx not implemented");
  }
};

template<typename Scalar, int NDim, int XDim, int TDim = XDim>
struct VectorFunction
{
  typedef Eigen::Matrix<Scalar, XDim, 1> VectorX;
  typedef Eigen::Matrix<Scalar, NDim, 1> VectorN;
  typedef Eigen::Matrix<Scalar, NDim, TDim> MatrixNT;

  virtual ~VectorFunction() = default;

  virtual void f(const VectorX& X, VectorN& f) = 0;

  virtual void f_fx(const VectorX& X, VectorN& f, MatrixNT& fx) = 0;
};
} // namespace mannumopt
