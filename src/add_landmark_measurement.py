import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    pose_4 = result.atPose2(X(4))
    landmark_2 = result.atPoint2(L(2))
    

    x_robot = pose_4.x()
    y_robot = pose_4.y()
    theta_robot = pose_4.theta()  
    
    x_land = landmark_2[0]
    y_land = landmark_2[1]
    
    dx = x_land - x_robot
    dy = y_land - y_robot

    distance = math.sqrt(dx**2 + dy**2)

    global_angle = math.atan2(dy, dx)
    bearing_radians = global_angle - theta_robot

    rotation = math.degrees(bearing_radians)
    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation), distance, MEASUREMENT_NOISE))
    return graph