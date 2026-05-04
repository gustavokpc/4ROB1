"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, follow_path
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439, 195) #439, 195
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        
        # Path planning variables
        self.exploration_iterations = 1000  # Number of iterations for exploration
        self.path = None
        self.path_index = 0
        self.phase = 'explore'  # Phase: 'explore', 'planning', or 'follow'

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp4()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        # self.tiny_slam.compute2() # Better compute

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()

        goal = [-200,-450,0]

        

        self.tiny_slam.update_map(self.lidar(), pose)
        self.occupancy_grid.display_cv(pose, goal=goal)
        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        print("\nPose: ", pose)
        print("\nForward command: ", command["forward"], "Rotation command: ", command["rotation"])
        print("\nGoal: ", goal)

        return command

    # def control_tp3(self):
    #         """
    #         Control function for TP3
    #         Map update and display with SLAM, path planning and path following
    #         """

    #     pose = self.odometer_values()
    #     goal = [-200,-400,0]
    #     self.tiny_slam.update_map(self.lidar(), pose)

    #     # Compute new command speed to perform obstacle avoidance
    #     command = potential_field_control(self.lidar(), pose, goal)

    #     print("\nPose: ", pose)
    #     print("\nForward command: ", command["forward"], "Rotation command: ", command["rotation"])
    #     print("\nGoal: ", goal)

    #     return command

    def control_tp4(self):
        """
        Control function for TP4
        Main control function with full SLAM, exploration, path planning and path following
        """
        pose = self.odometer_values()
        
        self.tiny_slam.localise(self.lidar(), pose)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
        self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
        
        # Phase 1: EXPLORATION
        if self.phase == 'explore':
            self.occupancy_grid.display_cv(self.corrected_pose)
            command = reactive_obst_avoid(self.lidar())
            self.counter += 1
            
            if self.counter >= self.exploration_iterations:
                print(f"\n=== Switching to planning phase after {self.counter} iterations ===")
                self.phase = 'planning'
        
        # Phase 2: PLANNING
        elif self.phase == 'planning':
            print(f"Current pose: {self.corrected_pose}")
            origin = np.array([0, 0, 0])
            self.path = self.planner.plan(self.corrected_pose, origin)
            
            if self.path is not None:
                print(f"Path found with {len(self.path[0])} waypoints")
                self.phase = 'follow'
                self.path_index = 0
                self.occupancy_grid.display_cv(self.corrected_pose, goal=origin, traj=self.path)
                command = reactive_obst_avoid(self.lidar())
            else:
                print("Failed to find path, returning to exploration")
                self.phase = 'explore'
                self.counter = 0
                command = reactive_obst_avoid(self.lidar())
        
        # Phase 3: FOLLOW PATH
        elif self.phase == 'follow':
            origin = np.array([0, 0, 0])
            command, self.path_index = follow_path(self.lidar(), self.corrected_pose, self.path, self.path_index)
            self.occupancy_grid.display_cv(self.corrected_pose, goal=origin, traj=self.path)
            
            # Check if reached the origin
            dist_to_origin = np.linalg.norm(self.corrected_pose[:2] - origin[:2])
            if dist_to_origin < 20:
                print(f"\n=== Reached origin! Distance: {dist_to_origin:.2f} ===")
                command = {"forward": 0.0, "rotation": 0.0}
                self.phase = 'explore'
                self.counter = 0
    
        return command
        