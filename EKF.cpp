#include <Eigen/Dense>

class EKF {
public:
    EKF();

    // Predict step of the EKF, using IMU data
    void Predict(const Eigen::Vector3f& angular_velocity,
        const Eigen::Vector3f& linear_acceleration,
        double dt);

    // Update step of the EKF, using odometry data
    void Update(const Eigen::Vector3f& position,
        const Eigen::Quaternionf& orientation);

private:
    // State vector: [position, velocity, orientation, angular velocity]
    Eigen::Matrix<float, 14, 1> state_;

    // State covariance matrix
    Eigen::Matrix<float, 14, 14> P_;
};

EKF::EKF() {
    // Initialize state and covariance
    state_.setZero();
    P_.setIdentity();
}

void EKF::Predict(const Eigen::Vector3f& angular_velocity,
    const Eigen::Vector3f& linear_acceleration,
    double dt) {
    // Extract current orientation from state vector
    Eigen::Quaternionf q(state_(6), state_(3), state_(4), state_(5));

    // Update position and velocity using linear acceleration
    state_.segment<3>(0) += dt * state_.segment<3>(3) + 0.5 * dt * dt * linear_acceleration;
    state_.segment<3>(3) += dt * linear_acceleration;

    // Convert angular velocity to quaternion
    Eigen::Quaternionf dq(1, 0.5 * dt * angular_velocity(0), 0.5 * dt * angular_velocity(1), 0.5 * dt * angular_velocity(2));

    // Update quaternion
    q = (q * dq).normalized();

    // Update orientation using angular velocity
    /*q.coeffs() += 0.5 * dt * q.coeffs() * Eigen::Vector4f(0, angular_velocity(0), angular_velocity(1), angular_velocity(2));
    q.normalize();*/
    state_.segment<4>(6) = q.coeffs();

    // Update angular velocity
    state_.segment<3>(10) = angular_velocity;

    // Compute Jacobian of motion model
    Eigen::Matrix<float, 14, 14> F;
    F.setIdentity();
    F.block<3, 3>(0, 3) = dt * Eigen::Matrix3f::Identity();
    F.block<3, 3>(3, 6) = -dt * q.toRotationMatrix();
    F.block<3, 3>(3, 10) = dt * Eigen::Matrix3f::Identity();

    // Update covariance using Jacobian and process noise
    Eigen::Matrix<float, 14, 14> Q;
    Q.setIdentity();
    Q *= 0.01; // Example process noise
    P_ = F * P_ * F.transpose() + Q;
}

void EKF::Update(const Eigen::Vector3f& position,
    const Eigen::Quaternionf& orientation) {
    // Compute measurement prediction
    Eigen::Vector3f z_pos = state_.segment<3>(0);
    Eigen::Quaternionf z_ori(state_(6), state_(3), state_(4), state_(5));
    Eigen::Vector3f z_vel = state_.segment<3>(3);
    Eigen::Vector3f z_ang_vel = state_.segment<3>(10);

    // Compute measurement Jacobian
    Eigen::Matrix<float, 6, 14> H;
    H.setZero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
    H.block<3, 3>(3, 6) = -z_ori.toRotationMatrix();
    Eigen::Matrix<float, 6, 6> R;
    R.setIdentity();
    R *= 0.01; // Example measurement noise

    // Compute Kalman gain
    Eigen::Matrix<float, 14, 6> K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();

    // Update state and covariance
    Eigen::Matrix<float, 6, 1> y;
    y << position - z_pos, orientation.coeffs() - z_ori.coeffs();
    state_ += K * y;
    P_ = (Eigen::Matrix<float, 14, 14>::Identity() - K * H) * P_;
}

int main() {
    EKF ekf;

    // Example IMU data
    Eigen::Vector3f angular_velocity(0.1, 0.2, 0.3);
    Eigen::Vector3f linear_acceleration(0.01, 0.02, 0.03);
    double dt = 0.1;

    // Example odometry data
    Eigen::Vector3f position(1, 2, 3);
    Eigen::Quaternionf orientation(1, 0, 0, 0);

    // Main loop
    while (true) {
        // Predict step
        ekf.Predict(angular_velocity, linear_acceleration, dt);

        // Update step
        ekf.Update(position, orientation);

        // Get current state
        // Eigen::Vector3f curr_pos = ekf.state().segment<3>(0);
        // Eigen::Quaternionf curr_ori(ekf.state()(6), ekf.state()(3), ekf.state()(4), ekf.state()(5));
        // Eigen::Vector3f curr_vel = ekf.state().segment<3>(3);
        // Eigen::Vector3f curr_ang_vel = ekf.state().segment<3>(10);

        // Do something with the current state, such as sending it to other modules or visualizing it
        // ...

        // Wait for next iteration
        // ...
    }
    return 0;
}