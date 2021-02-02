#pragma once

#include <mannumopt/fwd.hpp>

namespace mannumopt {
/// Modified Choleski decomposition for indefinite matrices.
/// See Algorithm 3.4 in Numerical Optimization of Nocedal and Wright
template<typename _MatrixType> class ApproxLDLT
{
  public:
    typedef _MatrixType MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef Eigen::Matrix<Scalar, RowsAtCompileTime, 1, 0, MaxRowsAtCompileTime, 1> TmpMatrixType;
    typedef Eigen::Index Index;
    typedef typename MatrixType::StorageIndex StorageIndex;

    /** \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via LDLT::compute(const MatrixType&).
      */
    ApproxLDLT()
      : m_matrix(),
        m_c(),
        m_beta(),
        m_delta(1e-8),
        m_isInitialized(false)
    {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa LDLT()
      */
    explicit ApproxLDLT(Index size)
      : m_matrix(size, size),
        m_c(size),
        m_isInitialized(false)
    {}

    /** \brief Constructor with decomposition
      *
      * This calculates the decomposition for the input \a matrix.
      *
      * \sa LDLT(Index size)
      */
    template<typename InputType>
    explicit ApproxLDLT(const Eigen::EigenBase<InputType>& matrix, bool evaluateBeta = true)
      : m_matrix(matrix.rows(), matrix.cols()),
        m_c(matrix.rows()),
        m_isInitialized(false)
    {
      compute(matrix.derived(), evaluateBeta);
    }

    inline const Eigen::TriangularView<const MatrixType, Eigen::UnitLower> matrixL() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return Eigen::TriangularView<const MatrixType, Eigen::UnitLower>(m_matrix);
    }

    inline Eigen::Diagonal<const MatrixType> vectorD() const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      return m_matrix.diagonal();
    }

    template<typename Rhs>
    inline const Eigen::Solve<ApproxLDLT, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "LDLT is not initialized.");
      eigen_assert(m_matrix.rows()==b.rows()
                && "LDLT::solve(): invalid number of rows of the right hand side matrix b");
      return Eigen::Solve<ApproxLDLT, Rhs>(*this, b.derived());
    }

    template<typename RhsType, typename DstType>
    EIGEN_DEVICE_FUNC
    void _solve_impl(const RhsType &rhs, DstType &dst) const
    {
      // L D L^T dst = b

      eigen_assert(rhs.rows() == rows());
      dst = rhs;

      // dst = L^-1 b
      matrixL().solveInPlace(dst);

      // dst = D^-1 (L^-1 b)
      dst.array() /= vectorD().array();

      // dst = L^-T (D^-1 L^-1 b)
      matrixL().transpose().solveInPlace(dst);
    }

    template<typename InputType>
    ApproxLDLT& compute(const Eigen::EigenBase<InputType>& matrix, bool evaluateBeta = true)
    {
      m_isInitialized = false;
      const Index size = rows();

      const InputType& m (matrix.derived());

      using std::max;
      using std::abs;

      if (evaluateBeta) {
        // Formula from Chap. 2, MODIFIED CHOLESKY DECOMPOSITION AND APPLICATIONS
        // by Thomas McSweeney
        // http://eprints.maths.manchester.ac.uk/2599/1/modified_cholesky_decomposition_and_applications.pdf
        Scalar ksi (abs(m(1,0)));
        for (Index i = 0; i < size; i++)
          for (Index j = 0; j < size; j++)
            if (i != j)
              ksi = max(ksi, abs(m(i,j)));
        Scalar gamma = m.diagonal().cwiseAbs().maxCoeff();

        using std::sqrt;
        m_beta = sqrt(max(max(gamma, ksi / sqrt(size*size-1)), m_delta));
      }

      m_matrix.resizeLike(matrix);
      m_c.resize(matrix.rows());

      auto d = m_matrix.diagonal();
      Scalar cjj, theta;
      for (Index j = 0; j < size-1; ++j) {
        cjj = m(j,j) - d.head(j).dot(m_matrix.row(j).head(j).cwiseAbs2());
        for (Index i = j+1; i < size; ++i) {
          m_c(i) = m(i,j) - (d.transpose().array() * m_matrix.row(i).array() * m_matrix.row(j).array()).head(j).sum();
          if (i == j+1) theta = abs(m_c(i));
          else theta = max(theta, abs(m_c(i)));
        }
        theta /= m_beta;
        Scalar dj = m_matrix(j,j) = max(max(abs(cjj), theta*theta), m_delta);
        for (Index i = j+1; i < size; ++i)
          m_matrix(i,j) = m_c(i) / dj;
      }
      Index j = size-1;
      cjj = m(j,j) - d.head(j).dot(m_matrix.row(j).head(j).cwiseAbs2());
      m_matrix(j,j) = max(abs(cjj), m_delta);

      m_isInitialized = true;

      return *this;
    }

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

  protected:
    MatrixType m_matrix;
    TmpMatrixType m_c;
    Scalar m_beta, m_delta;
    bool m_isInitialized;
};
}
