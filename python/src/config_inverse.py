# Configuration Inverse Kinematics
import numpy as np

from config_loader import load_parameters
rp = load_parameters("robot_parameters.py")

# Input angles
theta1 = 0; theta2 = 0; theta3 = 0; theta4 = 0

# DH parameters as a nested dictionary
DH_params_dict = {
    1: {'theta': np.radians(theta1),        'alpha': np.radians(90),    'd': rp.j1_height,  'r': 0.0},
    2: {'theta': np.radians(theta2 + 90),   'alpha': 0.0,               'd': 0.0,           'r': rp.link_a},
    3: {'theta': np.radians(theta3),        'alpha': 0.0,               'd': 0.0,           'r': rp.link_b},
    4: {'theta': np.radians(theta4),        'alpha': 0.0,               'd': 0.0,           'r': 43},
    5: {'theta': np.radians(-90),           'alpha': np.radians(-90),   'd': 0.0,           'r': 0.0}
}

def round_matrix(matrix, decimals=6):
    return np.round(matrix, decimals=decimals)

def zyx_rotation_matrix(alpha, beta, gamma):
    c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)
    c_beta, s_beta = np.cos(beta), np.sin(beta)
    c_gamma, s_gamma = np.cos(gamma), np.sin(gamma)
    
    R = np.array([
        [c_alpha * c_beta, c_alpha * s_beta * s_gamma - s_alpha * c_gamma, c_alpha * s_beta * c_gamma + s_alpha * s_gamma],
        [s_alpha * c_beta, s_alpha * s_beta * s_gamma + c_alpha * c_gamma, s_alpha * s_beta * c_gamma - c_alpha * s_gamma],
        [-s_beta, c_beta * s_gamma, c_beta * c_gamma]
    ])
    return R

def homogeneous_transform(alpha, beta, gamma, tx, ty, tz):
    # Get the 3x3 rotation matrix
    R = zyx_rotation_matrix( np.radians(alpha), np.radians(beta), np.radians(gamma) )
    
    # Create the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    return T

def dh_transform(theta, alpha, d, r):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), r*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                            1]
    ])