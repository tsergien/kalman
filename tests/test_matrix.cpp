#include "kf/matrix.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>

using kf::Matrix;

namespace {

template <std::size_t R, std::size_t C>
void expectNear(const Matrix<R, C>& a, const Matrix<R, C>& b, double tol = 1e-9) {
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << "at (" << i << ", " << j << ")";
}

// Detects whether A * B is a valid expression.
template <typename A, typename B>
concept Multipliable = requires(A a, B b) { a * b; };

template <typename M>
concept HasIdentity = requires { M::identity(); };

}  // namespace

// ---- compile-time checks ---------------------------------------------------

static_assert(Matrix<2, 3>::rows() == 2);
static_assert(Matrix<2, 3>::cols() == 3);
static_assert(Multipliable<Matrix<2, 3>, Matrix<3, 4>>);
static_assert(!Multipliable<Matrix<2, 3>, Matrix<2, 3>>, "shape mismatch must not compile");
static_assert(HasIdentity<Matrix<3, 3>>);
static_assert(!HasIdentity<Matrix<2, 3>>, "identity() only for square matrices");
static_assert(std::is_same_v<decltype(Matrix<2, 3>{}.transpose()), Matrix<3, 2>>);
static_assert(Matrix<2, 2>::identity()(1, 1) == 1.0, "identity() should be constexpr");
static_assert((Matrix<2, 2>::identity() * Matrix<2, 2>::identity())(0, 1) == 0.0);

// ---- runtime tests ---------------------------------------------------------

TEST(Matrix, DefaultIsZero) {
    Matrix<2, 3> m;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) EXPECT_EQ(m(i, j), 0.0);
}

TEST(Matrix, InitializerListIsRowMajor) {
    Matrix<2, 3> m{1, 2, 3,
                   4, 5, 6};
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_EQ(m(0, 2), 3.0);
    EXPECT_EQ(m(1, 0), 4.0);
    EXPECT_EQ(m(1, 2), 6.0);
}
TEST(Matrix, ElementWriteAccess) {
    Matrix<2, 2> m;
    m(1, 0) = 7.5;
    EXPECT_EQ(m(1, 0), 7.5);
}

TEST(Matrix, AddSubtract) {
    Matrix<2, 2> a{1, 2, 3, 4};
    Matrix<2, 2> b{10, 20, 30, 40};
    expectNear(a + b, Matrix<2, 2>{11, 22, 33, 44});
    expectNear(b - a, Matrix<2, 2>{9, 18, 27, 36});
}

TEST(Matrix, ScalarMultiplyBothSides) {
    Matrix<2, 2> a{1, 2, 3, 4};
    expectNear(2.0 * a, Matrix<2, 2>{2, 4, 6, 8});
    expectNear(a * 2.0, Matrix<2, 2>{2, 4, 6, 8});
}

TEST(Matrix, Transpose) {
    Matrix<2, 3> a{1, 2, 3, 4, 5, 6};
    expectNear(a.transpose(), Matrix<3, 2>{1, 4, 2, 5, 3, 6});
    expectNear(a.transpose().transpose(), a);
}

TEST(Matrix, MultiplyRectangular) {
    Matrix<2, 3> a{1, 2, 3,
        4, 5, 6};
        Matrix<3, 2> b{7, 8,
            9, 10,
            11, 12};
            expectNear(a * b, Matrix<2, 2>{58, 64, 139, 154});
        }
        
TEST(Matrix, IdentityIsNeutral) {
    Matrix<3, 3> a{2, -1, 0, -1, 2, -1, 0, -1, 2};
    expectNear(a * Matrix<3, 3>::identity(), a);
    expectNear(Matrix<3, 3>::identity() * a, a);
}


TEST(Matrix, TransposeOfProduct) {
    Matrix<2, 3> a{1, 2, 3, 4, 5, 6};
    Matrix<3, 2> b{0, 1, 1, 0, 2, -1};
    expectNear((a * b).transpose(), b.transpose() * a.transpose());
}

TEST(Matrix, Inverse2x2) {
    Matrix<2, 2> a{4, 7, 2, 6};
    expectNear(a.inverse(), Matrix<2, 2>{0.6, -0.7, -0.2, 0.4});
}

TEST(Matrix, InverseNeedsPivoting) {
    // a(0,0) == 0: naive Gauss-Jordan without pivoting divides by zero.
    Matrix<3, 3> a{0, 1, 2,
                   1, 0, 3,
                   4, -3, 8};
    expectNear(a * a.inverse(), Matrix<3, 3>::identity());
    expectNear(a.inverse() * a, Matrix<3, 3>::identity());
}

TEST(Matrix, InverseOfSingularThrows) {
    Matrix<2, 2> a{1, 2, 2, 4};
    EXPECT_THROW((void)a.inverse(), std::runtime_error);
}

TEST(Matrix, Cholesky2x2ByHand) {
    // [[4,2],[2,3]] = L L^T with L = [[2,0],[1,sqrt(2)]]
    Matrix<2, 2> a{4, 2, 2, 3};
    expectNear(a.cholesky(), Matrix<2, 2>{2, 0, 1, std::sqrt(2.0)});
}

TEST(Matrix, CholeskyReconstructs) {
    Matrix<3, 3> a{2, -1, 0, -1, 2, -1, 0, -1, 2};
    const auto L = a.cholesky();
    expectNear(L * L.transpose(), a);
    // lower triangular: zeros above the diagonal
    EXPECT_EQ(L(0, 1), 0.0);
    EXPECT_EQ(L(0, 2), 0.0);
    EXPECT_EQ(L(1, 2), 0.0);
}

TEST(Matrix, CholeskyOfNotPositiveDefiniteThrows) {
    // symmetric, but eigenvalues 3 and -1
    Matrix<2, 2> a{1, 2, 2, 1};
    EXPECT_THROW((void)a.cholesky(), std::runtime_error);
}

TEST(Matrix, Solve2x2ByHand) {
    Matrix<2, 2> a{4, 2, 2, 3};
    expectNear(a.solve(Matrix<2, 1>{8, 7}), Matrix<2, 1>{1.25, 1.5});
}

TEST(Matrix, SolveManyRightHandSidesMatchesInverse) {
    Matrix<3, 3> a{2, -1, 0, -1, 2, -1, 0, -1, 2};
    Matrix<3, 2> b{1, 4, 2, 5, 3, 6};
    const auto x = a.solve(b);
    expectNear(a * x, b);
    expectNear(x, a.inverse() * b);
}

TEST(Matrix, StreamOutputContainsAllElements) {
    Matrix<2, 2> a{1, 2, 3, 4};
    std::ostringstream os;
    os << a;
    const std::string s = os.str();
    for (const char* v : {"1", "2", "3", "4"}) EXPECT_NE(s.find(v), std::string::npos) << v;
}

