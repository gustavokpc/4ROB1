import random
import matplotlib.pyplot as plt
from gym_multigrid.multigrid import *
from gym_multigrid.core.agent import CollectActions, Agent,SmallActions
# Ensure Agent is explicitly imported
from gym_multigrid.core import *
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Wall, Goal,Lava
from gym_multigrid.core.world import World
# Ensure Goal is explicitly imported
# Add this at the top of your file with other imports
import enum

# Define your own custom actions enum that includes the 'available' attribute
class CustomSmallActions(enum.IntEnum):
    still = 0     # Stay in place
    left = 1      # Turn left
    right = 2     # Turn right
    forward = 3   # Move forward
    down = 4      # Move downward

    # Add this to satisfy the check in MultiGridEnv.step()
    def available(self):
        return ["build"]  # Return an iterable to avoid TypeError

# Then in your CustomMultiAgentEnv.__init__ method:

class CustomMultiAgentEnv(MultiGridEnv):
    """
    Multi-agent environment where agents navigate to their respective goal positions.
    
    Agents start at specified positions and must find paths to their designated goals.
    The environment supports adding obstacles (walls) and checking goal completion.
    """
    
    def __init__(
        self,
        width=10,
        height=10,
        max_steps=100,
        seed=None,
        agents=None
    ):
        # Default to 2 agents if not specified
        if agents is None:
            agents = [
                {"start": (1, 1), "goal": (width-2, height-2)},
                {"start": (2, 1), "goal": (width-2, 2)}
            ]
        
        self.agents_info = agents
        self.agent_goals = {}
        self.agent_start_dir = 0  # Default direction (right)
        
        # Initialize world with appropriate configuration
        from gym_multigrid.core.constants import COLORS
        # Define simplified OBJECT_TO_IDX for our problem
        OBJECT_TO_IDX = {
            "unseen": 0,
            "empty": 1,
            "wall": 2,
            "floor": 3,
            "goal": 8,
            "lava": 9,
            "agent": 10,
        }
        
        self.world = World(
            encode_dim=3,  # Default encoding dimension
            normalize_obs=False,
            OBJECT_TO_IDX=OBJECT_TO_IDX,
            COLORS=COLORS
        )
        # Create the agent objects based on the number of agents
        agents_list = [Agent(self.world) for _ in range(len(self.agents_info))]
        # Set default values for required parameters

        
        # Use CollectActions as the action set
        self.actions_set = CustomSmallActions
        see_through_walls = False
        partial_obs = False
        agent_view_size = 7
        render_mode = "rgb_array"
        uncached_object_types = []  # Initialize as empty list instead of None
        
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=agents_list,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )
        # Set seed if provided
        if seed is not None:
            self.seed(seed)
    def step(self, actions):
        """
        Fonction step complète gérant les mouvements des agents, 
        les collisions et les récompenses.
        """
        rewards = [0] * len(self.agents)
        terminated = False
        truncated = False
        infos = {"collisions": []}
        
        # Réinitialiser l'état de collision pour tous les agents
        for agent in self.agents:
            agent.collided = False
        
        # Calculer les positions prévues pour chaque agent après leur mouvement
        intended_positions = []
        for i, agent in enumerate(self.agents):
            if agent.terminated:
                intended_positions.append(tuple(agent.pos))
                continue
                
            action = actions[i]
            next_pos = list(agent.pos)
            
            if action == CustomSmallActions.still:
                pass
            elif action == CustomSmallActions.left or action == CustomSmallActions.right:
                # Les actions de rotation ne changent pas la position
                pass
            elif action == CustomSmallActions.forward:
                next_pos = agent.front_pos
            elif action == CustomSmallActions.down:
                # Position vers le sud
                next_pos = agent.south_pos()
            
            intended_positions.append(tuple(next_pos))
        
        # Détecter les collisions entre agents
        collisions_detected = [False] * len(self.agents)
        for i in range(len(self.agents)):
            if self.agents[i].terminated:
                continue
                
            # Vérifier si deux agents tentent d'occuper la même position
            for j in range(len(self.agents)):
                if i != j and not self.agents[j].terminated:
                    if intended_positions[i] == intended_positions[j]:
                        collisions_detected[i] = True
                        collisions_detected[j] = True
                        infos["collisions"].append((i, j))
        
        # Appliquer les actions et mettre à jour l'état collided
        for i, agent in enumerate(self.agents):
            if agent.terminated:
                continue
                
            # Marquer l'agent comme en collision si détecté
            agent.collided = collisions_detected[i]
            
            # Appliquer l'action uniquement si pas de collision
            # Sinon, l'agent reste sur place
            action = actions[i]
            
            if not collisions_detected[i]:
                if action == CustomSmallActions.still:
                    pass
                elif action == CustomSmallActions.left:
                    self._turn_left(i)
                elif action == CustomSmallActions.right:
                    self._turn_right(i)
                elif action == CustomSmallActions.forward:
                    self._move_forward(i, rewards)
                elif action == CustomSmallActions.down:
                    self._move_south(i, rewards)
            else:
                # Pénalité pour collision
                rewards[i] -= 5
        
        # Mettre à jour le compteur d'étapes
        self.step_count += 1
        
        # Vérifier les conditions de fin
        terminated = all(self.is_agent_at_goal(i) for i in range(len(self.agents)))
        truncated = self.step_count >= self.max_steps
        
        # Obtenir les observations
        obs = self._gen_obs()
        
        return obs, rewards, terminated, truncated, infos

    def _move_south(self, agent_idx, rewards):
        """Custom method to move agent south (down)"""
        agent = self.agents[agent_idx]
        current_dir = agent.dir
        
        # Face south (dir=1 in standard orientation)
        while agent.dir != 1:
            self._turn_right(agent_idx)
        
        # Attempt to move
        self._move_forward(agent_idx, rewards)
        
        # Restore original direction
        while agent.dir != current_dir:
            self._turn_left(agent_idx)

    def _turn_left(self, agent_idx):
        """Turn agent left"""
        agent = self.agents[agent_idx]
        agent.dir = (agent.dir - 1) % 4

    def _turn_right(self, agent_idx):
        """Turn agent right"""
        agent = self.agents[agent_idx]
        agent.dir = (agent.dir + 1) % 4

    def _move_forward(self, agent_idx, rewards):
        """Handle forward movement with lava checks"""
        agent = self.agents[agent_idx]
        fwd_pos = agent.front_pos
        
        # Get cell at target position
        fwd_cell = self.grid.get(*fwd_pos)
        
        # Check if movement is possible
        if fwd_cell is None or fwd_cell.can_overlap():
            # Check for lava
            if isinstance(fwd_cell, Lava):
                rewards[agent_idx] = -50
                agent.terminated = True
            else:
                # Move agent
                agent.pos = fwd_pos
                if tuple(agent.pos) == self.agent_goals[agent_idx]:
                    rewards[agent_idx] = 50
                    agent.terminated = True
                else:
                    rewards[agent_idx] = -1        
    # In your CustomMultiAgentEnv class, modify the _gen_grid method to create a custom Goal class
    def _gen_grid(self, width, height, world=None):
        # Create an empty grid using the imported Grid class
        self.grid = Grid(width, height, world=self.world)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Replace all walls with lava
        for x in range(width):
            for y in range(height):
                if isinstance(self.grid.get(x, y), Wall):
                    self.grid.set(x, y, Lava(self.world))
        
        # Create a custom Goal class that allows agents to enter
        class EnterableGoal(Goal):
            def can_enter(self):
                return True  # Allow agents to enter this cell
            # Add this method to the EnterableGoal class in your _gen_grid method:
            def can_overlap(self):
                return True  # Allow this object to be in the same cell as an agent
        
        # Place agents and goals
        for i, agent_info in enumerate(self.agents_info):
            # Get the agent already created by MultiGridEnv
            agent = self.agents[i]
            
            # Create and place agent
            start_pos = agent_info["start"]
            goal_pos = agent_info["goal"]
            self.place_agent(agent, start_pos)
            goal = EnterableGoal(self.world, index=i)  # Use our custom goal class
            self.grid.set(*goal_pos, goal)
            self.agent_goals[i] = goal_pos
        
    def add_wall(self, x, y):
        """Add a wall at the specified position using the imported Wall class"""
        if self.grid.get(x, y) is None:
            self.grid.set(x, y, Lava(self.world))
            return True
        return False
    
    def is_agent_at_goal(self, agent_idx):
        """Check if an agent has reached its goal"""
        agent = self.agents[agent_idx]
        agent_pos = tuple(agent.pos)  # Get position from the agent object
        goal_pos = self.agent_goals[agent_idx]
        return agent_pos == goal_pos
    def is_agent_in_lava(self, agent_idx, action=None):
        """
        Check if an agent is in lava or attempting to move into lava
        
        Parameters:
            agent_idx (int): The index of the agent to check
            action (int, optional): The action the agent is attempting to take (0-4)
            
        Returns:
            bool: True if agent is in lava or would move into lava, False otherwise
        """
        agent = self.agents[agent_idx]
        pos = agent.pos  # This is typically a tuple
        
        # If no action is specified, just check current position
        if action is None:
            return isinstance(self.grid.get(*pos), Lava)
        
        # Convert the tuple to a list for manipulation
        next_pos = list(pos)  # Convert tuple to list instead of using .copy()
        
        if action == 0:  # Stay in place
            return isinstance(self.grid.get(*pos), Lava)
        elif action == 2:  # Move east/right
            next_pos[0] += 1
        elif action == 4:  # Move south/down
            next_pos[1] += 1
        elif action == 1:  # Move west/left
            next_pos[0] -= 1
        elif action == 3:  # Move north/up
            next_pos[1] -= 1
        
        # Check if the next position contains lava
        next_cell = self.grid.get(next_pos[0], next_pos[1])
        return isinstance(next_cell, Lava)
    def are_agents_colliding(self, actions):
        """Vérifie les collisions entre agents basées sur leurs actions"""
        # Vérifier si les actions sont valides
        if len(actions) != len(self.agents):
            return [False] * len(self.agents)
        
        # Calculer les positions suivantes pour chaque agent
        next_positions = []
        for i, agent in enumerate(self.agents):
            # Copier la position actuelle
            next_pos = list(agent.pos)
            action = actions[i]
            
            # Calculer la nouvelle position selon l'action
            if action == CustomSmallActions.forward:
                next_pos = agent.front_pos
            elif action == CustomSmallActions.down:
                # Utiliser ta logique pour se déplacer vers le bas
                pass
                
            next_positions.append(tuple(next_pos))
        
        # Détecter les collisions
        collisions = [False] * len(self.agents)
        for i in range(len(self.agents)):
            for j in range(i+1, len(self.agents)):
                # Collision si deux agents tentent d'occuper la même cellule
                if next_positions[i] == next_positions[j]:
                    collisions[i] = True
                    collisions[j] = True
        
        return collisions, next_positions
    def all_agents_at_goals(self):
        """Check if all agents have reached their goals"""
        return all(self.is_agent_at_goal(i) for i in range(len(self.agents)))
    
    def visualize(self):
        """Visualize the environment using matplotlib"""
        grid_img = self.render(highlight=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img)
        plt.axis('off')
        plt.show()
    
    def run_random_agents(self, max_steps=100):
        """Run a simulation with random movements for all agents."""
        self.reset()
        self.visualize()  # Show the initial state
        
        for step in range(max_steps):
            if self.all_agents_at_goals():
                print(f"All agents reached their goals in {step} steps!")
                break
            
            # Generate random actions for all agents
            actions = []
            for i in range(len(self.agents)):
                if self.is_agent_at_goal(i):
                    actions.append(CustomSmallActions.still)  # Use the 'still' action when at the goal
                else:
                    # Exclude the 'down' action (4) if it's not implemented
                    valid_actions = [a for a in CustomSmallActions if a != CustomSmallActions.down]
                    actions.append(random.choice(valid_actions))  # Random action from valid actions
            
            # Step the environment with the generated actions
            _, _, terminated, truncated, _ = self.step(actions)
            
            
            
            # Check if the episode is done
            if terminated or truncated:
                print("Episode ended")
                break
        
        # Final visualization
        self.visualize()
