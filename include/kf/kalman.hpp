#pragma once

#include "kf/matrix.hpp"

namespace kf {

/**
 *      DICTIONARY
 *      N = state size
 *      x (m_x) - state vector (N×1) - physics we estimate
 *      P (m_P) - state covariance matrix (N×N) - diagonal elements are variances of state variables, off-diagonal elements are covariances between state variables
 *      F - state transition matrix (N×N)
 *      Q - process noise covariance matrix (N×N)  (How much do I distrust my prediction?)
 
 *      M = measurement size
 *      z - measurement vector (M×1) - what sensor measures
 *      H - measurement matrix (M×N) - converts the state into "sensor language"
 *      R - measurement noise covariance matrix (M×M) (How much do I distrust my sensor?)
 *  */ 


template<std::size_t N>
class KalmanFilter {
public:

    KalmanFilter(const Matrix<N, 1> &x0, const Matrix<N, N> &P0) : m_x(x0), m_P(P0) {}
    
    void predict(const Matrix<N, N> &F, const Matrix<N, N> &Q)
    {
        m_x = F * m_x;
        m_P = F * m_P * F.transpose() + Q;
        check_covariance();
    }

    template<std::size_t M> // M = sensor measurement size
    void update(const Matrix<M, 1> &z, const Matrix<M, N> &H, const Matrix<M, M> &R)
    {
        auto I = Matrix<N, N>::identity();
        auto PHt = m_P * H.transpose();

        auto y = z - H * m_x; // innovation
        auto S = H * PHt + R; // innovation covariance
        auto K = PHt * S.inverse(); // Kalman gain
                
        m_x += K * y;
        //P ← (I − KH) P (I − KH)ᵀ + K R Kᵀ
        auto IKH = I - K * H;
        m_P = IKH * m_P * IKH.transpose() + K * R * K.transpose();
        check_covariance();
    }

    [[nodiscard]] Matrix<N, 1> state() const { return m_x; }

    [[nodiscard]] Matrix<N, N> covariance() const { return m_P; }

private:
    /**
     * P(i, i) is the variance of state variable i.
     * a variance can never be negative: it's an average of squares.
     * if it's negative, the filter is broken: bad math, bad Q or R, or lost precision.
     * after that every estimate is garbage, but nothing crashes, so we'd never notice.
     * it also catches NaN: NaN >= 0 is false.
    */
    void check_covariance() const
    {
#ifdef __cpp_contracts
        // the loop is inside the lambda, not around contract_assert:
        // gcc 16.2 silently drops a contract_assert placed inside a loop (compiler bug).
        // "!(x >= 0)" instead of "x < 0" on purpose: it is true for NaN too.
        contract_assert([&] {
            for (std::size_t i = 0; i < N; ++i)
                if (!(m_P(i, i) >= 0.0)) return false;
            return true;
        }());
#endif
    }

    Matrix<N, 1> m_x{}; // state estimate
    Matrix<N, N> m_P{}; // estimate covariance
};

}  // namespace kf
