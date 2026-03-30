""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    laser_angle = lidar.get_ray_angles()

    if min(laser_dist) < 20:
        if laser_angle[np.argmin(laser_dist)] > 0:
            rotation_speed = -0.3
        else:
            rotation_speed = 0.3
        speed = 0.5
    else:
        rotation_speed = 0.0
        speed = 0.5

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    # ----------------------------------- Gradient repulsif -----------------------------------
    safe_distance = 25

    positive_laser_dist = np.where(lidar.get_sensor_values() > 0, lidar.get_sensor_values(), np.inf)
    distance_to_obstacle = np.min(positive_laser_dist)
    obstacle_angle = lidar.get_ray_angles()[np.argmin(lidar.get_sensor_values())]
    
    obstacle_position = current_pose[:2] + distance_to_obstacle * np.array([
        np.cos(current_pose[2] + obstacle_angle),
        np.sin(current_pose[2] + obstacle_angle)
    ])
    print("\n Distance to obstacle: ", distance_to_obstacle)
    if distance_to_obstacle > safe_distance:    
        gradient_repulsif = np.zeros(2)
    else:
        Kobs = 10000
        gradient_repulsif = - Kobs * (1/distance_to_obstacle - 1/safe_distance) * (obstacle_position - current_pose[:2]) / (distance_to_obstacle**3)
        print("\nGradient repulsif: ", gradient_repulsif)
        
    # ----------------------------------- Gradient atratif -----------------------------------
    distance_to_goal = np.linalg.norm(goal_pose[:2] - current_pose[:2])
    Kgoal = 1.0

    if distance_to_goal < 5:
        return {"forward": 0.0, "rotation": 0.0}

    gradient_atratif = Kgoal / distance_to_goal * (goal_pose[:2] - current_pose[:2])
    print("\nGradient atratif: ", gradient_atratif)

    # ----------------------------------- Gradient total -----------------------------------
    gradient_total = gradient_atratif + gradient_repulsif
    print("\nGradient total: ", gradient_total)

    K_omega = 1
    K_v = 1
    angle_to_goal = np.arctan2(gradient_total[1], gradient_total[0])
    omega_r = angle_to_goal - current_pose[2]
    omega_max = np.pi

    rotation_speed = K_omega*omega_r

    if abs(omega_r) < omega_max:
        forward_speed = K_v * np.linalg.norm(gradient_total)
    else:
        forward_speed = K_v * np.linalg.norm(gradient_total) * (omega_max / abs(omega_r))

    # rotation speed need to be between -1 and 1
    if rotation_speed > 1:
        rotation_speed = 1
    elif rotation_speed < -1:
        rotation_speed = -1

    # forward speed need to be between -1 and 1
    if forward_speed > 0.5:
        forward_speed = 0.5
    elif forward_speed < -0.5:
        forward_speed = -0.5

    print("\nDistance to goal: ", distance_to_goal)
   
    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command
