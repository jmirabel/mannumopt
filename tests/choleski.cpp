#include <mannumopt/decomposition/choleski.hpp>
#include <mannumopt/trust-region/two-dimensional.hpp>

#include <boost/test/unit_test.hpp>

using namespace mannumopt;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void run(MatrixXd M)
{
  ApproxLDLT<MatrixXd> ldlt (M);

  auto L = ldlt.matrixL().toDenseMatrix();
  auto D = ldlt.vectorD().eval().asDiagonal().toDenseMatrix();

  BOOST_CHECK(M.isApprox(L * D * L.transpose()));
  BOOST_TEST_MESSAGE('\n'
    << "M\n"<< M << '\n'
    << "L\n"<< L << '\n'
    << "D\n"<< D << '\n'
    << "L D L^T\n" << L * D * L.transpose() << '\n'
    << "M - L D L^T\n" << M - L * D * L.transpose());

  VectorXd b = VectorXd::Random(5);
  VectorXd x = ldlt.solve(b);
  BOOST_CHECK(b.isApprox(M*x));
  BOOST_TEST_MESSAGE('\n'
    << "x^T:     " << x.transpose() << '\n'
    << "(M x)^T: " << (M * x).transpose() << '\n'
    << "b^T:     " << b.transpose());
}

BOOST_AUTO_TEST_CASE(approx_choleski)
{
  MatrixXd M (5,5);
  M.setRandom();
  //TODO this cannot pass since M may not symmetric definite positive.
  //run(M);

  M = M * M.transpose();
  run(M);
}

