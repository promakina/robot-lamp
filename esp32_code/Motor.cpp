#include "Motor.h"
#include <algorithm> // For std::max and std::min

//extern SCSCL sc; // Make sure sc is defined and initialized in your main program

Motor::Motor(int center, int min_limit, int max_limit, int id, int direction)
    : center_(center), min_limit_(min_limit), max_limit_(max_limit), id_(id), direction_(direction) {}

int Motor::getCenter() const {
    return center_;
}

int Motor::getDirection() const {
    return direction_;
}

void Motor::setCenter(int center) {
    center_ = center;
}

void Motor::setDirection(int direction) {
    direction_ = direction;
}

int Motor::degree_to_byte(int degree) const{
// Total degrees the servo can rotate
  const int total_degrees = 300;
  // Total range of byte values
  const int byte_range = 1024;
  // Calculate the byte value per degree
  double bytes_per_degree = static_cast<double>(byte_range) / total_degrees;
  // Convert degree to byte value
  //double byte_value = degree * bytes_per_degree * direction_;
  double byte_value = degree * 3.11 * direction_; // Use experimental byte_per_degree value
  // Adjust for the MOTOR_CENTER calibration
  byte_value += center_;
  // Ensure the byte value is within the valid range [0, 1024]
  byte_value = std::max(0.0, std::min(static_cast<double>(byte_range), byte_value));
  
  return static_cast<int>(byte_value);
}

void Motor::driveToDegree(int degree, int t, int s, SCSCL& sc) {
    
    int desired_byte = degree_to_byte(degree);

    if (desired_byte <= max_limit_ && desired_byte >= min_limit_) {
        sc.WritePos(id_, desired_byte, t, s); // Call the SCServo function
    }

}
