#pragma once

#include <Eigen/Core>

#include <mannumopt/print.hpp>

#define MANNUMOPT_EIGEN_TYPEDEFS(Scalar,Dim)        \
  typedef Eigen::Matrix<Scalar, Dim, Dim> MatrixS;  \
  typedef Eigen::Matrix<Scalar, Dim, 1> VectorS;    \
  typedef Eigen::Matrix<Scalar, 1, Dim> RowVectorS

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
}

template<typename Scalar, int Dim>
struct Algo {
  MANNUMOPT_EIGEN_TYPEDEFS(Scalar, Dim);

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
}
