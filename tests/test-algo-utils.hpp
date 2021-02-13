#include <mannumopt/fwd.hpp>
#include <chrono>

template<int N>
struct Quadratic : mannumopt::VectorSpace<double, N> {
  MANNUMOPT_EIGEN_TYPEDEFS(double, N);

  MatrixS S;

  void f(const VectorS& X, double& f)
  {
    f = 0.5 * X.transpose() * S * X;
  }

  void f_fx(const VectorS& X, double& f, RowVectorS& fx)
  {
    this->f(X, f);
    fx = X.transpose() * S;
  }

  void f_fx_fxx(const VectorS& X, double& f, RowVectorS& fx, MatrixS& fxx)
  {
    f_fx(X, f, fx);
    fxx = S;
  }

  Quadratic() : S(MatrixS::Zero()) {
    for (int i = 0; i < N; ++i)
      S(i,i) = i+1;
  }
  Quadratic(VectorS d) : S(d.asDiagonal()) {}
  Quadratic(MatrixS S) : S(S) {}
};

template<int N>
struct Rosenbrock : mannumopt::VectorSpace<double, N> {
  MANNUMOPT_EIGEN_TYPEDEFS(double, N);

  double a = 100.;

  void f(const VectorS& X, double& f)
  {
    f = 0;
    for (int i = 0; i < N-1; ++i) {
      double x = X[i], y = X[i+1];

      double u = 1-x, v = y - x*x;
      f += u*u + a*v*v;
    }
  }

  void f_fx(const VectorS& X, double& f, RowVectorS& fx)
  {
    f = 0;
    fx.setZero();
    for (int i = 0; i < N-1; ++i) {
      double x = X[i], y = X[i+1];

      double u = 1-x, v = y - x*x;
      f += u*u + a*v*v;

      fx[i  ] += - 2 * u - 4 * a * x * v;
      fx[i+1] += 2 * a * v;
    }
  }

  void f_fx_fxx(const VectorS& X, double& f, RowVectorS& fx, MatrixS& fxx)
  {
    f = 0;
    fx.setZero();
    fxx.setZero();
    for (int i = 0; i < N-1; ++i) {
      double x = X[i],
             y = X[i+1];

      double u = 1 - x,
             v = y - x*x;

      f += u*u + a*v*v;

      fx[i  ] += - 2*u - 4*a*x*v;
      fx[i+1] += 2*a*v;

      fxx(i,i) += 2 - 4*a*(y - 3*x*x);
      fxx(i,i+1) += -4*a*x;
      fxx(i+1,i) += -4*a*x;
      fxx(i+1,i+1) += 2*a;
    }
  }

  constexpr Rosenbrock(double a) : a(a) {}
};

void status(auto what, auto algo, bool success, auto start, auto end, auto f, auto x)
{
  namespace chrono = std::chrono;

  double v;
  f.f(x,v);

  BOOST_CHECK(success);
  BOOST_TEST_MESSAGE(
      (success ? (v > 0.1 ? "[warn]":"[ ok ]") : "[fail]" )
      << ' ' << what << " (" << algo.iter << "): f(" << x.transpose() << ") = " << v
      << " in " << chrono::duration_cast<chrono::microseconds>(end - start).count() << "us");
}

#define _SINGLE_TEST(algo,size,function,...) BOOST_AUTO_TEST_CASE(algo ## _ ## function ## _ ## size) { test_##algo<size, function>(__VA_ARGS__); }
#define TEST_ALGO(algo)                         \
  _SINGLE_TEST(algo,4,Rosenbrock, 1.)           \
  _SINGLE_TEST(algo,6,Rosenbrock, 1.)           \
  _SINGLE_TEST(algo,8,Rosenbrock, 1.)           \
  _SINGLE_TEST(algo,4,Quadratic)               \
  _SINGLE_TEST(algo,6,Quadratic)               \
  _SINGLE_TEST(algo,8,Quadratic)
