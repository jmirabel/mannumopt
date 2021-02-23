#include <chrono>
#include <map>

#include <mannumopt/fwd.hpp>
#include <mannumopt/function.hpp>

template<int N>
struct Quadratic : mannumopt::Function<double, N, N> {
  MANNUMOPT_EIGEN_TYPEDEFS(double, N, S);

  MatrixS A;
  VectorS B;
  double c = 0;

  void f(const VectorS& X, double& f)
  {
    f = 0.5 * X.transpose() * A * X + B.dot(X) + c;
  }

  void f_fx(const VectorS& X, double& f, RowVectorS& fx)
  {
    this->f(X, f);
    fx = X.transpose() * A + B.transpose();
  }

  void f_fx_fxx(const VectorS& X, double& f, RowVectorS& fx, MatrixS& fxx)
  {
    f_fx(X, f, fx);
    fxx = A;
  }

  Quadratic(int n = N) : A(MatrixS::Zero(n,n)), B(VectorS::Zero(n)) {
    for (int i = 0; i < N; ++i)
      A(i,i) = i+1;
  }
  Quadratic(MatrixS A) : A(A), B(VectorS::Zero(A.rows())) {}
};

template<int N>
struct Rosenbrock : mannumopt::Function<double, N, N> {
  MANNUMOPT_EIGEN_TYPEDEFS(double, N, S);

  double a = 100.;

  void f(const VectorS& X, double& f)
  {
    int n = (int)X.size();
    f = 0;
    for (int i = 0; i < n-1; ++i) {
      double x = X[i], y = X[i+1];

      double u = 1-x, v = y - x*x;
      f += u*u + a*v*v;
    }
  }

  void f_fx(const VectorS& X, double& f, RowVectorS& fx)
  {
    int n = (int)X.size();
    f = 0;
    fx.setZero();
    for (int i = 0; i < n-1; ++i) {
      double x = X[i], y = X[i+1];

      double u = 1-x, v = y - x*x;
      f += u*u + a*v*v;

      fx[i  ] += - 2 * u - 4 * a * x * v;
      fx[i+1] += 2 * a * v;
    }
  }

  void f_fx_fxx(const VectorS& X, double& f, RowVectorS& fx, MatrixS& fxx)
  {
    int n = (int)X.size();
    f = 0;
    fx.setZero();
    fxx.setZero();
    for (int i = 0; i < n-1; ++i) {
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

bool only_warn = false;
struct Statistics {
  int nsuccess;
  int ntotal;
};
std::map<std::string, Statistics> stats;

void status(auto what, auto algo, bool success, auto start, auto end, auto f, auto x,
    double v)
{
  namespace chrono = std::chrono;

  Statistics& stat = stats[what];

  stat.ntotal++;
  if (success) stat.nsuccess++;

  if (only_warn) BOOST_WARN (success);
  else           BOOST_CHECK(success);
  BOOST_TEST_MESSAGE(
      (success ? (v > 0.1 ? "[warn]":"[ ok ]") : "[fail]" )
      << ' ' << what << " (" << algo.iter << "): f(" << x.transpose() << ") = " << v
      << " in " << chrono::duration_cast<chrono::microseconds>(end - start).count() << "us");
}

void status(auto what, auto algo, bool success, auto start, auto end, auto f, auto x)
{
  double v;
  f.f(x,v);
  status(what, algo, success, start, end, f, x, v);
}

void print_statistics(const char* header = nullptr)
{
  if (header != NULL) BOOST_TEST_MESSAGE(header);
  for (const auto& pair : stats) {
    BOOST_TEST_MESSAGE(pair.first
        << ": " << pair.second.nsuccess << " / " << pair.second.ntotal);
  }
}

int verbosityLevel() {
  const auto& master_test_suite = boost::unit_test::framework::master_test_suite();
  static int level = -1;
  if (level < 0) {
    level = 0;
    for (int i = 0; i < master_test_suite.argc; ++i)
      if (strcmp("--verbose", master_test_suite.argv[i]) == 0
          || strcmp("-v", master_test_suite.argv[i]) == 0)
        level++;
  }
  return level;
}

#define _SINGLE_TEST(algo,warn,size,function,...) \
  BOOST_AUTO_TEST_CASE(algo ## _ ## function ## _ ## size) { \
    stats.clear();                                           \
    only_warn = warn;                                        \
    test_##algo<size, function>(__VA_ARGS__);                \
    print_statistics(#algo "_" #function "_" #size);          \
  }
#define TEST_ALGO(algo)                         \
  _SINGLE_TEST(algo,true ,4,Rosenbrock, 1.)           \
  _SINGLE_TEST(algo,true ,6,Rosenbrock, 1.)           \
  _SINGLE_TEST(algo,true ,8,Rosenbrock, 1.)           \
  _SINGLE_TEST(algo,false,4,Quadratic,4)              \
  _SINGLE_TEST(algo,false,6,Quadratic,6)              \
  _SINGLE_TEST(algo,false,8,Quadratic,8)
