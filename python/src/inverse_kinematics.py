'''
Position of JOINT 4 ONLY

'''
import numpy as np
import math
import config_inverse
import logging
import warnings

def inverse_kin(tx, ty, tz):
    # Input XYZ are Joint 4 location
    # Tool Frame: Translations & Rotations (degrees)
    tfx, tfy, tfz = 0.0, 0.0, 0.0
    tfRz, tfRy, tfRx = 0.0, 0.0, 0.0
    
    # Calculate J1
    if (tx ==0):
        if(ty > 0): j1 = 90
        elif (ty < 0): j1 = -90
        else: j1 = 0
    else:
        j1 = np.rad2deg( np.arctan(ty/tx) )
    
    # Rotate by -j1 and find new points
    xp = tx*np.cos(np.radians(-j1)) - ty*np.sin(np.radians(-j1))
    yp = 0.0 # rotate to J1 = 0
    zp = tz # No need to rate Z

    # Get the homogeneous transformation matrix
    # 0-T
    rx , ry, rz = 0, 0 , 0 # No rotation along X and Z
    R_0_T = config_inverse.homogeneous_transform(rx, ry, rz, xp, yp, zp)
    
    # Tool Frame
    TF  = config_inverse.homogeneous_transform(tfRz, tfRy, tfRx, tfx, tfy, tfz)
    
    # Tool Frame inverted matrix
    TF_inv = np.linalg.inv(TF)

    # Find R 0-5
    R_0_3 = np.dot(R_0_T,TF_inv)

    # X,Z coordinates of Joint 4
    _tx = R_0_3[0,3]
    _tz = R_0_3[2,3]

    # Now verify that Joint 4 can reach _tx and _tz
    # Looking in 2D (x,z) only since already rotated by J1
    d1 = config_inverse.DH_params_dict[1]['d']
    r2 = config_inverse.DH_params_dict[2]['r']
    r3 = config_inverse.DH_params_dict[3]['r']
    arm_length = r2 + r3 # J2 to J4
    point_j2 = np.array([ 0 , d1 ]) # Point J2
    target_point = np.array([ _tx, _tz ])
    target_vector_length = np.linalg.norm(target_point - point_j2)
    if( target_vector_length > arm_length):
        return round(j1,3), math.nan, math.nan
    # L1 and L2
    L1 = abs(_tz - d1 )
    L2 = np.sqrt(_tx**2 + L1**2)

    # Theta B
    if( _tx == 0 ): theta_b = np.radians(90)
    else: theta_b = np.arctan(abs(L1/_tx))
    #elif (xp > 0): theta_b = np.atan(L1/xp)
    #else: theta_b = np.radians(180) - np.atan(abs(L1/xp))

    theta_b = np.rad2deg(theta_b)

    # Theta C
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning) #convert the warning to an error.
            theta_c = np.arccos( (r2**2 + L2**2 - r3**2) / (2*r2*L2) )
        theta_c = np.rad2deg(theta_c)
    except Exception as e:
        logging.error(f'Error: {e}')
        logging.warning(f'Vals, r2 {r2}, L2 {L2}, r3 {r3}')
        logging.warning('Returning NaN')
        return math.nan, math.nan, math.nan

    # Calculate J2
    if( _tx < 0): j2 = 90 + theta_c - theta_b
    elif( _tz < d1 ): j2 = theta_c - theta_b - 90
    else: 
        if( (theta_b + theta_c) < 90): j2 = theta_b + theta_c - 90
        else: j2 = theta_b + theta_c - 90

    # Theta D, formed by triangle r2, r2, L2
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning) #convert the warning to an error.
            theta_d = np.arccos( (r2**2 + r3**2 - L2**2) / (2*r2*r3) )
        theta_d = np.rad2deg(theta_d)
    except Exception as e:
        logging.warning(f'Error: {e}')
        logging.warning(f'Vals, r2 {r2}, L2 {L2}, r3 {r3}')
        logging.warning('Returning NaN')
        return math.nan, math.nan, math.nan

    # J3
    j3 = theta_d - 180

    return round(j1,3), round(j2,3), round(j3,3)

if __name__ == "__main__":
    print("Running Directly")
    x = 14.05
    y = 3.99
    z = 264.58
    #j1, j2, j3 = inverse_kin(x,y,z)
    these_angles = inverse_kin(x,y,z)
    #print(f"Angles: J1: {j1}, J2: {j2}, J3: {j3}")
    print(f"Angles: {these_angles}")







    

