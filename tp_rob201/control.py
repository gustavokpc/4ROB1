""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Desvio de obstáculos com margem de segurança lateral.
    safe_dist: distância mínima desejada de qualquer parede.
    """
    laser_dist = lidar.get_sensor_values()
    laser_angle = lidar.get_ray_angles()
    
    # 1. Definimos o que é uma direção livre
    free_dist_threshold = 100 
    
    # 2. Encontrar o melhor caminho (maior distância)
    idx_max_dist = np.argmax(laser_dist)
    best_angle = laser_angle[idx_max_dist]
    max_val_found = laser_dist[idx_max_dist]

    # 3. Verificar a distância à frente (zona de colisão direta)
    front_mask = np.abs(laser_angle) < np.radians(10) # Aumentei um pouco para 10º
    dist_front = np.min(laser_dist[front_mask])

    # 4. LÓGICA DE DISTÂNCIA DE SEGURANÇA (Parede lateral)
    # Encontramos o ponto mais próximo de qualquer lugar ao redor do robô
    idx_min_dist = np.argmin(laser_dist)
    min_dist_found = laser_dist[idx_min_dist]
    closest_angle = laser_angle[idx_min_dist]

    # INICIALIZAÇÃO DE COMANDOS
    speed = 0.4
    rotation_speed = 0.0
    safe_dist = 5  # Distância de segurança para paredes laterais

    # LÓGICA DE DECISÃO
    if min_dist_found < safe_dist:
        # PRIORIDADE 1: Se algo estiver MUITO perto (zona de segurança), 
        # gire para o lado oposto ao obstáculo mais próximo.
        speed = 0.1
        # Se o obstáculo está na direita (ângulo negativo), gira para esquerda (positivo)
        # Se está na esquerda (ângulo positivo), gira para direita (negativo)
        rotation_speed = -0.4 * np.sign(closest_angle)
        
    elif dist_front < free_dist_threshold:
        # PRIORIDADE 2: Se a frente não está livre, busca o melhor ângulo
        speed = 0.1
        rotation_speed = 0.2 * np.sign(best_angle)
        
        # Caso de beco sem saída (melhor ângulo é frente, mas frente < threshold)
        if np.abs(best_angle) < 0.05 and max_val_found < free_dist_threshold:
            rotation_speed = 0.2
    
    else:
        # Caminho livre à frente e longe de paredes
        speed = 0.4
        rotation_speed = 0.0

    return {"forward": speed, "rotation": rotation_speed}


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
    safe_distance = 10

    positive_laser_dist = np.where(lidar.get_sensor_values() > 0, lidar.get_sensor_values(), np.inf)
    distance_to_obstacle = np.min(positive_laser_dist)
    obstacle_angle = lidar.get_ray_angles()[np.argmin(lidar.get_sensor_values())]
    
    obstacle_position = current_pose[:2] + distance_to_obstacle * np.array([
        np.cos(current_pose[2] + obstacle_angle),
        np.sin(current_pose[2] + obstacle_angle)
    ])
    # print("\n Distance to obstacle: ", distance_to_obstacle)
    if distance_to_obstacle > safe_distance:    
        gradient_repulsif = np.zeros(2)
    else:
        Kobs = 10000
        gradient_repulsif = - Kobs * (1/distance_to_obstacle - 1/safe_distance) * (obstacle_position - current_pose[:2]) / (distance_to_obstacle**3)
        # print("\nGradient repulsif: ", gradient_repulsif)
        
    # ----------------------------------- Gradient atratif -----------------------------------
    distance_to_goal = np.linalg.norm(goal_pose[:2] - current_pose[:2])
    Kgoal = 1.0

    if distance_to_goal < 5:
        return {"forward": 0.0, "rotation": 0.0}

    gradient_atratif = Kgoal / distance_to_goal * (goal_pose[:2] - current_pose[:2])
    # print("\nGradient atratif: ", gradient_atratif)

    # ----------------------------------- Gradient total -----------------------------------
    gradient_total = gradient_atratif + gradient_repulsif
    # print("\nGradient total: ", gradient_total)

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
    if rotation_speed > 0.6:
        rotation_speed = 0.6
    elif rotation_speed < -0.6:
        rotation_speed = -0.6

    # forward speed need to be between -1 and 1
    if forward_speed > 0.3:
        forward_speed = 0.3
    elif forward_speed < -0.3:
        forward_speed = -0.3

    print("\nDistance to goal: ", distance_to_goal)
    print("\nGoal: ", goal_pose[:2])
   
    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command


def follow_path(lidar, current_pose, path, current_index):
    """
    Follow a planned path
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose
    path : trajectory as [x, y] array
    current_index : index of current target point in path
    """
    # If reached the end, stop
    if current_index >= len(path[0]):
        print("\n=== Reached final target! ===")
        return {"forward": 0.0, "rotation": 0.0}, current_index

    # Get next target point
    target_x = path[0][current_index]
    target_y = path[1][current_index]
    target = np.array([target_x, target_y, 0])

    # Distance to target
    dist_to_target = np.linalg.norm(target[:2] - current_pose[:2])

    # If close to target, move to next
    if dist_to_target < 10:  # 10 units threshold
        current_index += 1
        if current_index >= len(path[0]):
            return {"forward": 0.0, "rotation": 0.0}, current_index
        target_x = path[0][current_index]
        target_y = path[1][current_index]
        target = np.array([target_x, target_y, 0])
        print("\nCurrent index: ", current_index)

    # Use potential field towards target
    return potential_field_control(lidar, current_pose, target), current_index
