#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt {
template<typename Scalar, int XDim, int TDim = XDim>
struct Function
{
  MANNUMOPT_ALGO_TYPEDEFS(Scalar, XDim, TDim);

  typedef Scalar     Output;
  typedef RowVectorT Derivative;
  typedef MatrixTT   Hessian;

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

  typedef VectorN  Output;
  typedef VectorN  OutputVector;
  typedef MatrixNT Derivative;

  virtual ~VectorFunction() = default;

  virtual int dimension() = 0;

  inline void residual(const VectorX& X, double& r)
  {
    VectorN f(dimension());
    this->f(X, f);
    r = .5 * f.squaredNorm();
  }

  virtual void f(const VectorX& X, VectorN& f) = 0;

  virtual void f_fx(const VectorX& X, VectorN& f, MatrixNT& fx) = 0;
};
} // namespace mannumopt
