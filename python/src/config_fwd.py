# Configuration
import numpy as np

def dh_transform(theta, alpha, d, r):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), r*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                            1]
    ])

def rotation_matrix_zyx(alpha, beta, gamma):
    # Rotation around Z-axis
    R_z = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
    
    # Rotation around Y-axis
    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    
    # Rotation around X-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])
    
    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    return R

def homogeneous_transform(alpha, beta, gamma, tx, ty, tz): # Assemble homogeneous transform
    
    # Get the 3x3 rotation matrix
    R = rotation_matrix_zyx(alpha, beta, gamma)
    
    # Create the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    return T

def round_matrix(matrix, decimals=6):
    return np.round(matrix, decimals=decimals)

def extract_zyx_euler_angles(T):
    # Ensure the matrix is in a valid range for arctan2 operations
    if T[2, 0] > 1.0:
        T[2, 0] = 1.0
    if T[2, 0] < -1.0:
        T[2, 0] = -1.0
    
    phi = np.arctan2(-T[2, 0], np.sqrt(T[0, 0]**2 + T[1, 0]**2)) # Y axis
    
    if np.cos(phi) == 0:
        theta = 0 # Z axis
        psi = np.arctan2(T[0, 1], T[1, 1]) # X axis
    else:
        theta = np.arctan2(T[1, 0] / np.cos(phi), T[0, 0] / np.cos(phi)) # Z axis
        psi = np.arctan2(T[2, 1] / np.cos(phi), T[2, 2] / np.cos(phi)) # X axis
    
    return np.rad2deg(psi), np.rad2deg(phi), np.rad2deg(theta)

