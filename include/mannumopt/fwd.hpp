#pragma once

#include <Eigen/Core>

#include <mannumopt/print.hpp>

#define MANNUMOPT_EIGEN_TYPEDEFS(Scalar,Dim,Suffix)        \
  typedef Eigen::Matrix<Scalar, Dim, Dim> Matrix##Suffix;  \
  typedef Eigen::Matrix<Scalar, Dim, 1> Vector##Suffix;    \
  typedef Eigen::Matrix<Scalar, 1, Dim> RowVector##Suffix

namespace mannumopt {

typedef int size_type;

namespace internal
{
  template<typename Scalar>
  inline void scalar_addition(Scalar& x_plus_v, const Scalar& x, const Scalar& v) { x_plus_v = x + v; }

  template<typename Scalar, int Dim>
  inline void vector_space_addition(Eigen::Matrix<Scalar, Dim, 1>& x_plus_v,
      const Eigen::Matrix<Scalar, Dim, 1>& x, const Eigen::Matrix<Scalar, Dim, 1>& v)
  {
    x_plus_v = x + v;
  }

  template<typename T> struct traits {};
}

template<typename Scalar, int XDim, int TDim = XDim>
struct Algo {
  typedef Eigen::Matrix<Scalar, TDim, TDim> MatrixTT;
  typedef Eigen::Matrix<Scalar, TDim, 1> VectorT;
  typedef Eigen::Matrix<Scalar, 1, TDim> RowVectorT;

  typedef Eigen::Matrix<Scalar, XDim, 1> VectorX;
  typedef Eigen::Matrix<Scalar, 1, XDim> RowVectorX;

  Scalar xtol, fxtol2;
  size_type maxIter = 200;
  std::ostream* cout = nullptr;

  size_type iter;

  inline bool verbose() { return (cout != nullptr); }

  template <typename ...Args>
  inline void print(bool print_headers, const Args&... args)
  {
    if (verbose()) mannumopt::print(*cout, print_headers, args...);
  }
};

template<typename Scalar, int XDim, int TDim = XDim> struct BFGS;
template<typename Scalar, int XDim, int TDim = XDim> struct NewtonTR;
template<typename Scalar, int XDim, int TDim = XDim> struct NewtonLS;
template<typename Scalar, int XDim, int TDim = XDim> struct GaussNewton;

#define MANNUMOPT_ALGO_TYPEDEFS(Scalar,XDim,TDim)    \
  typedef Algo<Scalar, XDim, TDim> AlgoBase;         \
  typedef typename AlgoBase::MatrixTT   MatrixTT;    \
  typedef typename AlgoBase::VectorT    VectorT;     \
  typedef typename AlgoBase::RowVectorT RowVectorT;  \
  typedef typename AlgoBase::VectorX    VectorX;     \
  typedef typename AlgoBase::RowVectorX RowVectorX
}
