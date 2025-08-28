#ifndef MOTOR_H
#define MOTOR_H

#include <SCServo.h>

class Motor {
public:
    Motor(int center, int min_limit, int max_limit, int id, int direction);

    int getCenter() const;
    int getDirection() const;

    void setCenter(int center);
    void setDirection(int direction);

    int degree_to_byte(int degree) const;
    void driveToDegree(int degree, int t, int s, SCSCL& sc); // Drive servo

private:
    int center_;
    int min_limit_;
    int max_limit_;
    int id_;
    int direction_; // Direction of degree-to-byte conversion

};

#endif // MOTOR_H
