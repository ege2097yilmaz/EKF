cmake_minimum_required(VERSION 3.10)
project(EKF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find Eigen library
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Create executable
add_executable(ekf EKF.cpp ${EKF_SOURCES})

# Link Eigen library
target_link_libraries(ekf Eigen3::Eigen)