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
    }

    [[nodiscard]] Matrix<N, 1> state() const { return m_x; }

    [[nodiscard]] Matrix<N, N> covariance() const { return m_P; }

private:
    Matrix<N, 1> m_x{}; // state estimate
    Matrix<N, N> m_P{}; // estimate covariance
};

}  // namespace kf
