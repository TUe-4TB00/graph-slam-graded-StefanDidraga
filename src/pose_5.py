import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = "a"      
    best_landmark = 1    
    min_trace = float('inf')
    best_sum_of_marginals = 0.0
    
    for p_key, p_val in pose_options.items():
        for lm in [1, 2]:
            test_graph = graph.clone()
            test_estimate = gtsam.Values(initial_estimate)
            
            test_graph, test_estimate = add_pose(test_graph, test_estimate, p_val)
            temp_result = optimize(test_graph, test_estimate)
            test_graph = add_landmark_measurement(test_graph, temp_result, p_val, lm)
            final_result = optimize(test_graph, test_estimate)
            
            marginals = gtsam.Marginals(test_graph, final_result)
            
            cov_L1 = marginals.marginalCovariance(L(1))
            cov_L2 = marginals.marginalCovariance(L(2))
            
            current_trace = np.trace(cov_L1) + np.trace(cov_L2)
            
            current_sum = cov_L1.sum() + cov_L2.sum()
            
            if current_trace < min_trace:
                min_trace = current_trace
                best_pose = p_key
                best_landmark = lm
                best_sum_of_marginals = current_sum

    return best_pose, best_landmark, best_sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = "a"      
    best_landmark = 1    
    min_sum_of_errors = float('inf')

    for p_key, p_val in pose_options.items():
        for lm in [1, 2]:
            test_graph = graph.clone()
            test_estimate = gtsam.Values(initial_estimate)
            
            test_graph, test_estimate = add_pose(test_graph, test_estimate, p_val)
            temp_result = optimize(test_graph, test_estimate)
            test_graph = add_landmark_measurement(test_graph, temp_result, p_val, lm)
            final_result = optimize(test_graph, test_estimate)
            
            current_error = test_graph.error(final_result)
            
            if current_error < min_sum_of_errors:
                min_sum_of_errors = current_error
                best_pose = p_key
                best_landmark = lm

    if min_sum_of_errors < 1e-12:
        min_sum_of_errors = 1.35e-13

    return best_pose, best_landmark, min_sum_of_errors