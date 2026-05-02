""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        score = 0

        lidar_distances = lidar.get_sensor_values()
        lidar_angles = lidar.get_ray_angles()
        # close_points = lidar_distances < 400
        # lidar_distances = lidar_distances[close_points]
        # lidar_angles = lidar_angles[close_points]

        # Convert lidar readings to world coordinates
        x_laser_world = pose[0] + lidar_distances * np.cos(pose[2] + lidar_angles)
        y_laser_world = pose[1] + lidar_distances * np.sin(pose[2] + lidar_angles)
        
        # Convert laser points to map coordinates
        x_laser_map, y_laser_map = self.grid.conv_world_to_map(x_laser_world, y_laser_world)
        # Filter out points that are outside the map boundaries
        select = np.logical_and(np.logical_and(x_laser_map >= 0, x_laser_map < self.grid.x_max_map),
                                np.logical_and(y_laser_map >= 0, y_laser_map < self.grid.y_max_map))
        x_laser_map = x_laser_map[select]
        y_laser_map = y_laser_map[select]
        # print("Filtered laser points in map coordinates:", len(x_laser_map))

        # Add value to the map for the laser end points
        score = np.sum(self.grid.occupancy_map[x_laser_map, y_laser_map])
        return score    

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
        d = np.linalg.norm(odom_pose[:2] - odom_pose_ref[:2])
        alpha = np.arctan2(odom_pose[1] - odom_pose_ref[1], odom_pose[0] - odom_pose_ref[0])
        corrected_pose = [odom_pose_ref[0] + d*np.cos(odom_pose_ref[2] + alpha), odom_pose_ref[1] + d*np.sin(odom_pose_ref[2] + alpha), odom_pose[2] + odom_pose_ref[2]]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        absolute_pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, absolute_pose)
        best_ref = self.odom_pose_ref.copy()
        sigma = np.array([1, 1, np.radians(10)])
        N = 100
        trials_without_improvement = 0

        while trials_without_improvement < N:
            # Generate a random pose around the current reference pose
            random_ref = best_ref + np.random.normal(0, sigma, size=3)
            
            # Compute the score for the random pose
            score = self._score(lidar, self.get_corrected_pose(raw_odom_pose, random_ref))

            # Update the best score and reference pose if the new score is better
            if score > best_score:
                best_score = score
                best_ref = random_ref
                trials_without_improvement = 0  # Reset counter if we found a better pose
            else:
                trials_without_improvement += 1  # Increment counter if no improvement
        self.odom_pose_ref = best_ref
        # print("\nBest score: ", best_score)
        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """

        # Convert lidar points to world coordinates
        ranges = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()

        # Valid ranges and corresponding angles  
        valid = np.logical_and(np.isfinite(ranges), ranges > 0)
        if not np.any(valid):
            return
        ranges = ranges[valid]
        ray_angles = ray_angles[valid]

        points_x = pose[0] + ranges * np.cos(pose[2] + ray_angles)
        points_y = pose[1] + ranges * np.sin(pose[2] + ray_angles)

        # points_x_minus1 = pose[0] + (ranges - 1) * np.cos(pose[2] + ray_angles)
        # points_y_minus1 = pose[1] + (ranges - 1) * np.sin(pose[2] + ray_angles)

        # points_x_minus2 = pose[0] + (ranges - 2) * np.cos(pose[2] + ray_angles)
        # points_y_minus2 = pose[1] + (ranges - 2) * np.sin(pose[2] + ray_angles)

        # points_x_minus3 = pose[0] + (ranges - 3) * np.cos(pose[2] + ray_angles)
        # points_y_minus3 = pose[1] + (ranges - 3) * np.sin(pose[2] + ray_angles)        

        # points_x_minus4 = pose[0] + (ranges - 4) * np.cos(pose[2] + ray_angles)
        # points_y_minus4 = pose[1] + (ranges - 4) * np.sin(pose[2] + ray_angles)

        # points_x_minus5 = pose[0] + (ranges - 5) * np.cos(pose[2] + ray_angles)
        # points_y_minus5 = pose[1] + (ranges - 5) * np.sin(pose[2] + ray_angles)

        # points_x_minus6 = pose[0] + (ranges - 6) * np.cos(pose[2] + ray_angles)
        # points_y_minus6 = pose[1] + (ranges - 6) * np.sin(pose[2] + ray_angles)

        # # Add points to lines between robot and lidar points
        # for x, y in zip(points_x_minus6, points_y_minus6):
        #     self.grid.add_value_along_line(pose[0], pose[1], x, y, val=-1)
        
        # self.grid.add_map_points(points_x_minus5, points_y_minus5, val=-1)
        # self.grid.add_map_points(points_x_minus4, points_y_minus4, val=-1)
        # self.grid.add_map_points(points_x_minus3, points_y_minus3, val=0)
        # self.grid.add_map_points(points_x_minus2, points_y_minus2, val=2)
        # self.grid.add_map_points(points_x_minus1, points_y_minus1, val=3)
        # self.grid.add_map_points(points_x, points_y, val=4)

        points_x_minus1 = pose[0] + (ranges - 1) * np.cos(pose[2] + ray_angles)
        points_y_minus1 = pose[1] + (ranges - 1) * np.sin(pose[2] + ray_angles)

        points_x_minus2 = pose[0] + (ranges - 2) * np.cos(pose[2] + ray_angles)
        points_y_minus2 = pose[1] + (ranges - 2) * np.sin(pose[2] + ray_angles)

        points_x_minus3 = pose[0] + (ranges - 3) * np.cos(pose[2] + ray_angles)
        points_y_minus3 = pose[1] + (ranges - 3) * np.sin(pose[2] + ray_angles)

        # Add points to lines between robot and lidar points
        for x, y in zip(points_x_minus3, points_y_minus3):
            self.grid.add_value_along_line(pose[0], pose[1], x, y, val=-1)
        
        self.grid.add_map_points(points_x_minus2, points_y_minus2, val=3)
        self.grid.add_map_points(points_x_minus1, points_y_minus1, val=4)
        self.grid.add_map_points(points_x, points_y, val=5)

        np.clip(self.grid.occupancy_map, -40, 40, out=self.grid.occupancy_map)
        # Add points to the map
        # TODO for TP3

    # def compute(self):
    #     """ Useless function, just for the exercise on using the profiler """
    #     # Remove after TP1

    #     ranges = np.random.rand(3600)
    #     ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

    #     # Poor implementation of polar to cartesian conversion
    #     points = []
    #     for i in range(3600):
    #         pt_x = ranges[i] * np.cos(ray_angles[i])
    #         pt_y = ranges[i] * np.sin(ray_angles[i])
    #         points.append([pt_x, pt_y])

    # def compute2(self):
    #     """ Useless function, just for the exercise on using the profiler """
    #     # Remove after TP1

    #     ranges = np.random.rand(3600)
    #     ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)


    #     points_x = ranges * np.cos(ray_angles)
    #     points_y = ranges * np.sin(ray_angles)


        
