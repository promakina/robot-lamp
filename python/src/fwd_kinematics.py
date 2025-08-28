'''
Find XYZ of Joint 4
'''
import numpy as np
import config_fwd

from config_loader import load_parameters
rp = load_parameters("robot_parameters.py")

def fwd_kin(theta1, theta2, theta3, theta4):
    """
    Performs forward kinematics

    Inputs: 
    joint angles

    Returns:
    XYZ coordinates
    """
    # DH parameters as a nested dictionary
    DH_params_dict = {
        1: {'theta': np.radians(theta1),        'alpha': np.radians(90),    'd': rp.j1_height,  'r': 0.0},
        2: {'theta': np.radians(theta2 + 90),   'alpha': 0.0,               'd': 0.0,           'r': rp.link_a},
        3: {'theta': np.radians(theta3),        'alpha': 0.0,               'd': 0.0,           'r': rp.link_b},
        4: {'theta': np.radians(theta4),        'alpha': 0.0,               'd': 0.0,           'r': 43.0},
        5: {'theta': np.radians(-90),           'alpha': np.radians(-90),   'd': 0.0,           'r': 0.0}
    }

    transformation_matrices = {}  # Dictionary to store transformation matrices

    # Create transformation matrices for each joint
    for joint, params in DH_params_dict.items():
        theta = params['theta']
        alpha = params['alpha']
        d = params['d']
        r = params['r']
        
        transform_matrix = config_fwd.dh_transform(theta, alpha, d, r)
        transformation_matrices[joint] = transform_matrix  # Save the matrix
    
    # Transformation matrices from 0
    zero_matrices = {}
    for joint, matrix in transformation_matrices.items():
        if joint == 1:
            prev_matrix = matrix
        else:
            zero_matrices[joint] = np.dot(prev_matrix,matrix)
            prev_matrix = zero_matrices[joint]

    # Tool Frame
    # Angles in radians
    alpha = np.radians(0)  # Rotation around Z-axis
    beta = np.radians(0)   # Rotation around Y-axis
    gamma = np.radians(0)  # Rotation around X-axis
    # Translations (offsets)
    tx, ty, tz = 0.0, 0.0, 0.0  # No offsets for this example

     # Get the homogeneous transformation matrix
    T = config_fwd.homogeneous_transform(alpha, beta, gamma, tx, ty, tz)

    #T_0 = config_fwd.round_matrix(np.dot(zero_matrices[4],T))
    T_0 = np.dot(zero_matrices[3],T)

    # Extract ZYX Euler angles
    psi, phi, theta = config_fwd.extract_zyx_euler_angles(T_0)
    # Extract XYZ
    x, y, z = T_0[0,3], T_0[1,3], T_0[2,3]
    
    return round(x,3), round(y,3), round(z,3)  # Return position coordinates of Joint 4

if __name__ == "__main__":  
    print ("Running Directly") 
    a = [0,0,0,0]
    x,y,z = fwd_kin(*a)
    #x,y,z = fwd_kin(0,0,0,0)
    print(f"Coordinates: X: {x}, Y: {y}, Z: {z} ")
