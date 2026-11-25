#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace kf {
    
template<std::size_t R, std::size_t C, typename T = double>
class Matrix {
public:
    constexpr Matrix() = default;
    
    constexpr Matrix(std::initializer_list<T> elements) {
        if (elements.size() != R*C) throw std::invalid_argument("initializer_list size must be R*C");
        
        std::copy(elements.begin(), elements.end(), m_data.begin());
    }

    static constexpr std::size_t rows() {return R;}
    static constexpr std::size_t cols() {return C;}
    
    constexpr T& operator()(std::size_t i, std::size_t j) { return m_data[i*C + j]; }
    constexpr const T& operator()(std::size_t i, std::size_t j) const { return m_data[i*C + j];}
    
    constexpr Matrix& operator+=(const Matrix& rhs) {
        for (std::size_t i = 0; i < R * C; ++i)
        {
            this->m_data[i] += rhs.m_data[i];
        }
        return *this;
    }

    [[nodiscard]] friend constexpr Matrix 
    operator+(Matrix lhs, const Matrix& rhs)
    {
        lhs += rhs;
        return lhs;
    }

    constexpr Matrix& operator-=(const Matrix& rhs) {
        for (std::size_t i = 0; i < R * C; ++i)
        {
            this->m_data[i] -= rhs.m_data[i];
        }
        return *this;
    }

    [[nodiscard]] friend constexpr Matrix 
    operator-(Matrix lhs, const Matrix& rhs)
    {
        lhs -= rhs;
        return lhs;
    }

    constexpr Matrix& operator*=(T a) {
        for (std::size_t i = 0; i < R * C; ++i)
        {
            this->m_data[i] *= a;
        }
        return *this;
    }
    [[nodiscard]] friend constexpr Matrix operator*(Matrix lhs, T a)
    {
        lhs *= a;
        return lhs;
    }
    [[nodiscard]] friend constexpr Matrix operator*(T a, Matrix rhs)
    {
        rhs *= a;
        return rhs;
    }

    [[nodiscard]] constexpr Matrix<C, R, T> transpose() const
    {
        Matrix<C, R, T> result;
        for (std::size_t i = 0; i < R; ++i)
        {
            for (std::size_t j = 0; j < C; ++j)
                result(j, i) = m_data[i*C + j];
        }
        return result;
    }

    [[nodiscard]] static constexpr Matrix<R, C, T> identity() requires (R == C)
    {
        Matrix result;
        for (std::size_t i = 0; i < R; ++i)
            result(i, i) = T{1};
        return result;
    }

    // (R×C)·(C×K) -> R×K
    template<std::size_t K>
    [[nodiscard]] friend constexpr Matrix<R, K, T> operator*(const Matrix<R, C, T> &lhs, const Matrix<C, K, T> &rhs)
    {
        Matrix<R, K, T> result;
        // loops are ordered for 'cache-friendliness' so no 'jumps' in memory access
        for (std::size_t r = 0; r < R; ++r)
        {
            for (std::size_t c = 0; c < C; ++c)
            {
                for (std::size_t k = 0; k < K; ++k)
                    result(r, k) += lhs(r, c) * rhs(c, k);
            }
        }
        return result;
    }

    // AA^{-1} = I, A^{-1}A = I. if det(A) == 0 inverse doesn't exist.
    // for every column i find pivot row, swap thia row to be at pos i, scale, eliminate. O(n^3). Gauss-Jordan with partial pivoting.
    [[nodiscard]] constexpr Matrix<R, C, T> inverse() const requires (R == C)
    {
        auto A = *this; // copy of the original matrix
        auto inv = Matrix<R, C, T>::identity();

        for (std::size_t r = 0; r < R; ++r)
        {
            // find pivot row
            std::size_t pivot_row = r;
            for (std::size_t i = r + 1; i < R; ++i)
            {
                if (std::abs(A(i, r)) > std::abs(A(pivot_row, r)))
                    pivot_row = i;
            }

            // swap pivot row with current row
            if (pivot_row != r)
            {
                for (std::size_t j = 0; j < C; ++j)
                {
                    std::swap(A(r, j), A(pivot_row, j));
                    std::swap(inv(r, j), inv(pivot_row, j));
                }
            }

            // check for singularity
            if (std::abs(A(r, r)) < 1e-9)
                throw std::runtime_error("Matrix is singular and cannot be inverted.");

            // scale pivot row : divide all elements by pivot to make it 1
            T scale = A(r, r);
            for (std::size_t j = 0; j < C; ++j)
            {
                A(r, j) /= scale;
                inv(r, j) /= scale;
            }

            // eliminate: make 0 all other rows in column r
            for (std::size_t i = 0; i < R; ++i)
            {
                if (i != r)
                {
                    T factor = A(i, r);
                    for (std::size_t j = 0; j < C; ++j)
                    {
                        A(i, j) -= factor * A(r, j);
                        inv(i, j) -= factor * inv(r, j);
                    }
                }
            }
        }
        return inv;
    }

    // A = L L^T, L lower triangular with positive diagonal. Exists only if A is symmetric positive definite.
    // column by column: L(j,j) = sqrt(A(j,j) - sum L(j,k)^2), L(i,j) = (A(i,j) - sum L(i,k) L(j,k)) / L(j,j), k < j. O(n^3 / 3).
    // reads only the lower triangle of A, symmetry is assumed, not checked.
    [[nodiscard]] constexpr Matrix<R, C, T> cholesky() const requires (R == C)
    {
        const auto& A = *this; // no copy, we only read
        Matrix<R, C, T> L;

        for (std::size_t j = 0; j < R; ++j)
        {
            // diagonal element
            T diag = A(j, j);
            for (std::size_t k = 0; k < j; ++k)
                diag -= L(j, k) * L(j, k);

            // sqrt of a number <= 0: A is not positive definite (in KF: broken P or R)
            if (diag <= T{0})
                throw std::runtime_error("Matrix is not positive definite, Cholesky failed.");
            // std::sqrt can run at compile time (constexpr) since c++26
            L(j, j) = std::sqrt(diag);

            // elements below the diagonal in column j
            for (std::size_t i = j + 1; i < R; ++i)
            {
                T sum = A(i, j);
                for (std::size_t k = 0; k < j; ++k)
                    sum -= L(i, k) * L(j, k);
                L(i, j) = sum / L(j, j);
            }
        }
        return L;
    }

    // solves A X = B for X without computing A^{-1}. A must be symmetric positive definite.
    // A = L L^T, so L (L^T X) = B: forward substitution L U = B, then back substitution L^T X = U.
    // every column of B is solved independently.
    template<std::size_t K>
    [[nodiscard]] constexpr Matrix<R, K, T> solve(const Matrix<R, K, T>& B) const requires (R == C)
    {
        const auto L = cholesky();

#ifdef __cpp_contracts
        // we divide by L(i, i) below, so it must not be zero.
        // cholesky() already promises every L(i, i) > 0 (it throws otherwise).
        // the user can't break this, only a bug in cholesky() can.
        // that's why it's a contract_assert and not an exception.
        contract_assert([&] {
            for (std::size_t i = 0; i < R; ++i)
                if (!(L(i, i) > T{0})) return false;
            return true;
        }());
#endif

        // forward substitution: top to bottom, row i has one new unknown U(i, c)
        Matrix<R, K, T> U;
        for (std::size_t c = 0; c < K; ++c)
        {
            for (std::size_t i = 0; i < R; ++i)
            {
                T sum = B(i, c);
                for (std::size_t k = 0; k < i; ++k)
                    sum -= L(i, k) * U(k, c);
                U(i, c) = sum / L(i, i);
            }
        }

        // back substitution: bottom to top, L^T(i, k) = L(k, i) so no transpose needed
        Matrix<R, K, T> X;
        for (std::size_t c = 0; c < K; ++c)
        {
            // i goes R-1 ... 0. size_t can't be negative, so check first, then decrement
            for (std::size_t i = R; i-- > 0;)
            {
                T sum = U(i, c);
                for (std::size_t k = i + 1; k < R; ++k)
                    sum -= L(k, i) * X(k, c);
                X(i, c) = sum / L(i, i);
            }
        }
        return X;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<R, C, T>& m)
    {
        for (std::size_t i = 0; i < R; ++i)
        {
            for (std::size_t j = 0; j < C; ++j)
            {
                os << m(i, j);
                if (j < C - 1) os << " ";
            }
            if (i < R - 1) os << "\n";
        }
        return os;
    }
    
private:
    std::array<T, R * C> m_data{};
};


}  // namespace kf
