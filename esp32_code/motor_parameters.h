#ifndef MOTOR_PARAMETERS_H
#define MOTOR_PARAMETERS_H
 
#include "Motor.h"

// Define center, direction values for 6 motors
// Center value in byte
const int MOTOR1_CENTER = 392;
const int MOTOR1_DIRECTION = -1;

const int MOTOR2_CENTER = 492;
const int MOTOR2_DIRECTION = 1;

const int MOTOR3_CENTER = 469;
const int MOTOR3_DIRECTION = -MOTOR2_DIRECTION;

const int MOTOR4_CENTER = 520;
const int MOTOR4_DIRECTION = -1;

const int MOTOR5_CENTER = 350;
const int MOTOR5_DIRECTION = 1;

// Define max, min limits
// Values in byte

const int MOTOR_MIN = 20;
const int MOTOR_MAX = 1003;

// Create instances of the Motor class for each motor
Motor motor1(MOTOR1_CENTER, MOTOR_MIN, MOTOR_MAX, 1, MOTOR1_DIRECTION);
Motor motor2(MOTOR2_CENTER, MOTOR_MIN, MOTOR_MAX, 2, MOTOR2_DIRECTION);
Motor motor3(MOTOR3_CENTER, MOTOR_MIN, MOTOR_MAX, 3, MOTOR3_DIRECTION);
Motor motor4(MOTOR4_CENTER, MOTOR_MIN, MOTOR_MAX, 4, MOTOR4_DIRECTION);
Motor motor5(MOTOR5_CENTER, MOTOR_MIN, MOTOR_MAX, 5, MOTOR5_DIRECTION);

#endif // MOTOR_PARAMETERS_H
