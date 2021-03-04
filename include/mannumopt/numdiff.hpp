#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt {
namespace numdiff {

namespace internal {
template<typename A, typename B, bool scalar = std::is_same<typename A::Scalar, B>::value > struct assign_tpl;

template<typename A, typename B> struct assign_tpl<A,B,true > { static inline void run(A& a, const B& b) { a[0] = b; } };
template<typename A, typename B> struct assign_tpl<A,B,false> { static inline void run(A& a, const B& b) { a    = b; } };

template<typename A, typename B> inline void assign(A&& a, const B& b) { assign_tpl<A, B>::run(a, b); }
}

template<typename Function, typename Integrate, typename InputVector, typename OutputDerivative>
void forward(Function& func, const Eigen::MatrixBase<InputVector>& x,
    const typename InputVector::Scalar eps,
    const Eigen::MatrixBase<OutputDerivative>& fx_out,
    Integrate integrate)
{
  typedef Eigen::Matrix<typename OutputDerivative::Scalar, OutputDerivative::ColsAtCompileTime, 1> InputTangent;
  typedef typename Function::Output OutputVector;

  OutputDerivative& fx = const_cast<OutputDerivative&>(fx_out.derived());
  OutputVector v_x, v_x_t;
  InputTangent t (InputTangent::Zero(fx.cols()));
  InputVector x_t (x);

  func.f(x, v_x);

  for (int i = 0; i < fx.cols(); ++i) {
    t[i] = eps;
    integrate(x_t, x, t);
    func.f(x_t, v_x_t);
    internal::assign(fx.col(i), (v_x_t - v_x) / eps);

    t[i] = 0;
  }
}

template<typename Function, typename InputVector, typename OutputDerivative>
void forward(Function& func, const Eigen::MatrixBase<InputVector>& x,
    const typename InputVector::Scalar eps,
    const Eigen::MatrixBase<OutputDerivative>& fx_out)
{
  auto integrate = &mannumopt::internal::vector_space_addition<typename InputVector::Scalar, InputVector::RowsAtCompileTime>;
  forward(func, x, eps, fx_out, integrate);
}

template<typename Function, typename Integrate, typename InputVector, typename OutputDerivative>
void central(Function& func, const Eigen::MatrixBase<InputVector>& x,
    const typename InputVector::Scalar eps,
    const Eigen::MatrixBase<OutputDerivative>& fx_out,
    Integrate integrate)
{
  typedef Eigen::Matrix<typename OutputDerivative::Scalar,
    OutputDerivative::ColsAtCompileTime, 1> InputTangent;
  typedef typename Function::Output OutputVector;

  OutputDerivative& fx = const_cast<OutputDerivative&>(fx_out.derived());
  OutputVector v_x_m_t, v_x_p_t;
  InputTangent t (InputTangent::Zero(fx.cols()));
  InputVector x_t (x);

  for (int i = 0; i < fx.cols(); ++i) {
    t[i] = -eps/2;
    integrate(x_t, x, t);
    func.f(x_t, v_x_m_t);

    t[i] = eps/2;
    integrate(x_t, x, t);
    func.f(x_t, v_x_p_t);

    internal::assign(fx.col(i), (v_x_p_t - v_x_m_t) / eps);
    t[i] = 0;
  }
}

template<typename Function, typename InputVector, typename OutputDerivative>
void central(Function& func, const Eigen::MatrixBase<InputVector>& x,
    const typename InputVector::Scalar eps,
    const Eigen::MatrixBase<OutputDerivative>& fx_out)
{
  auto integrate = &mannumopt::internal::vector_space_addition<typename InputVector::Scalar, InputVector::RowsAtCompileTime>;
  central(func, x, eps, fx_out, integrate);
}

template<typename Function, typename Integrate, typename InputVector, typename OutputDerivative>
void second_order_central(Function& func, const Eigen::MatrixBase<InputVector>& x,
    const typename InputVector::Scalar eps,
    const Eigen::MatrixBase<OutputDerivative>& fx_out,
    Integrate integrate)
{
  typedef Eigen::Matrix<typename OutputDerivative::Scalar,
    OutputDerivative::ColsAtCompileTime, 1> InputTangent;
  typedef typename Function::Output OutputVector;

  OutputDerivative& fx = const_cast<OutputDerivative&>(fx_out.derived());
  OutputVector v_x;
  InputTangent t (InputTangent::Zero(fx.cols()));
  InputVector x_t (x);

  func.f(x, v_x);

  fx.diagonal().array() = -2*v_x;
  auto eps2 = eps*eps;

  for (int i = 0; i < fx.cols(); ++i) {
    t[i] = -eps;
    integrate(x_t, x, t);
    func.f(x_t, v_x);
    fx(i,i) += v_x;

    t[i] = eps;
    integrate(x_t, x, t);
    func.f(x_t, v_x);
    fx(i,i) += v_x;

    fx(i,i) /= eps2;

    for (int j = i+1; j < fx.cols(); ++j) {
      t[i] = eps/2;

      t[j] = eps/2;
      integrate(x_t, x, t);
      func.f(x_t, v_x);
      fx(i,j) = v_x;

      t[j] = -eps/2;
      integrate(x_t, x, t);
      func.f(x_t, v_x);
      fx(i,j) -= v_x;

      t[i] = -eps/2;

      t[j] = eps/2;
      integrate(x_t, x, t);
      func.f(x_t, v_x);
      fx(i,j) -= v_x;

      t[j] = -eps/2;
      integrate(x_t, x, t);
      func.f(x_t, v_x);
      fx(i,j) += v_x;

      fx(i,j) /= eps2;
      fx(j,i) = fx(i,j);

      t[j] = 0;
    }
    t[i] = 0;
  }
}

template<typename Function, typename InputVector, typename OutputDerivative>
void second_order_central(Function& func, const Eigen::MatrixBase<InputVector>& x,
    const typename InputVector::Scalar eps,
    const Eigen::MatrixBase<OutputDerivative>& fx_out)
{
  auto integrate = &mannumopt::internal::vector_space_addition<typename InputVector::Scalar, InputVector::RowsAtCompileTime>;
  second_order_central(func, x, eps, fx_out, integrate);
}

} // namespace numdiff
} // namespace mannumopt
