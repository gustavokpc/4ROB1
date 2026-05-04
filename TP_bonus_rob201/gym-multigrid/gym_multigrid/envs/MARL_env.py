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
    The environment supports adding obstacles (walls/lava) and checking goal completion.
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
        self.lava_positions = set()
        self.prev_positions = {}
        
        # Track objects under agents to restore goals when agents move
        self.objects_under_agents = {}
        
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
        
        # Use CustomSmallActions as the action set
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
        if seed is not None:
            self.seed(seed)
            
        self.prev_positions = {}

    def _gen_grid(self, width, height, world=None):
        self.grid = Grid(width, height, world=self.world)
        
        self.grid.wall_rect(0, 0, width, height)
        
        class EnterableLava(Lava):
            def can_enter(self):
                return True  
        
        self.lava_positions = set()  
        for x in range(width):
            for y in range(height):
                if isinstance(self.grid.get(x, y), Wall):
                    self.grid.set(x, y, EnterableLava(self.world))
                    self.lava_positions.add((x, y))  # Stocker la position de la lave
        
        # Create a custom Goal class that allows agents to enter
        class EnterableGoal(Goal):
            def can_enter(self):
                return True  # Allow agents to enter this cell
            
            def can_overlap(self):
                return True  # Allow this object to be in the same cell as an agent
        
        # List of distinct colors to use for agents and goals
        agent_colors = ['red', 'blue', 'green', 'purple', 'yellow', 'orange']
        
        # Initialize the objects_under_agents dictionary
        self.objects_under_agents = {}
        
        # Place agents and goals with matching colors
        for i, agent_info in enumerate(self.agents_info):
            # Use a color from the list (cycling if needed)
            color_idx = i % len(agent_colors)
            color = agent_colors[color_idx]
            
            # Set agent color
            agent = self.agents[i]
            agent.color = color
            
            # Place the agent at the start position
            start_pos = agent_info["start"]
            self.place_agent(agent, start_pos)
            
            # Create goal with the same color and place it
            goal_pos = agent_info["goal"]
            goal = EnterableGoal(self.world, index=i)
            goal.color = color  # Set the same color as the agent
            self.grid.set(*goal_pos, goal)
            self.agent_goals[i] = goal_pos

    def add_wall(self, x, y):
        """Ajoute un mur de lave à la position spécifiée"""
        if 0 <= x < self.width and 0 <= y < self.height:
            class EnterableLava(Lava):
                def can_enter(self):
                    return True
            
            current_obj = self.grid.get(x, y)
            if current_obj:
                self.grid.set(x, y, None)
            
            lava = EnterableLava(self.world)
            self.grid.set(x, y, lava)
            self.lava_positions.add((x, y))  
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
        Vérifie si un agent est sur une cellule de lave.
        
        Parameters:
            agent_idx (int): L'indice de l'agent à vérifier
            action (int, optional): L'action que l'agent tente d'effectuer
            
        Returns:
            bool: True si l'agent est sur une cellule de lave, False sinon
        """
        agent = self.agents[agent_idx]
        if agent.pos is None:
            return False
        
        return tuple(agent.pos) in self.lava_positions
    
    def _move_agent(self, action, agent_idx):
        """
        Déplace un agent en fonction de l'action donnée.
        Si l'agent tente de sortir de la grille, l'action est convertie en "stay".
        
        Args:
            action (int): Action à exécuter (0-4)
            agent_idx (int): Indice de l'agent à déplacer
        
        Returns:
            bool: True si l'agent a effectivement bougé, False sinon
        """
        agent = self.agents[agent_idx]
        if agent.pos is None:
            return False
        
        if action == 0:  # Stay in place
            return False
        
        pos = list(agent.pos)
        next_pos = pos.copy()
        
        if action == 2:  # Move east/right
            next_pos[0] += 1
        elif action == 4:  # Move south/down
            next_pos[1] += 1
        elif action == 1:  # Move west/left
            next_pos[0] -= 1
        elif action == 3:  # Move north/up
            next_pos[1] -= 1
        else:
            return False  
        
        if next_pos[0] < 0 or next_pos[0] >= self.width or next_pos[1] < 0 or next_pos[1] >= self.height:
            # Si hors limites, considérer comme "stay" et ne pas bouger
            return False
        
        next_cell = self.grid.get(next_pos[0], next_pos[1])
        
        if next_cell is not None and hasattr(next_cell, 'can_enter') and not next_cell.can_enter():
            return False
        
        if agent_idx in self.objects_under_agents:
            self.grid.set(pos[0], pos[1], self.objects_under_agents[agent_idx])
        else:
            self.grid.set(pos[0], pos[1], None)
        
        self.objects_under_agents[agent_idx] = next_cell
        
        # Mettre à jour la position de l'agent
        agent.pos = next_pos
        self.grid.set(next_pos[0], next_pos[1], agent)
        
        return True  # L'agent a bougé
    def step(self, actions):
        """
        Exécute une étape de l'environnement avec les actions données pour chaque agent.

        Args:
            actions: Liste d'actions, une par agent

        Returns:
            observations: Liste des observations pour chaque agent
            rewards: Liste des récompenses pour chaque agent
            terminated: Si l'épisode est terminé par atteinte des objectifs
            truncated: Si l'épisode est tronqué par dépassement du nombre max d'étapes
            info: Informations supplémentaires
        """
        # Vérifier que actions est une liste avec le bon nombre d'éléments
        if not isinstance(actions, list):
            actions = [actions]

        if len(actions) != len(self.agents):
            actions = actions + [0] * (len(self.agents) - len(actions))

        # Vérifier les collisions potentielles
        collisions = self.are_agents_colliding()

        # Exécuter les actions pour chaque agent
        moves = []
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            # Ne pas bouger si collision détectée
            if not collisions[i]:
                moved = self._move_agent(action, i)
                moves.append(moved)
            else:
                moves.append(False)

        # Incrémenter le compteur d'étapes
        self.step_count += 1

        # Générer les observations
        # Get just the positions of all agents as the state
        state = [tuple(agent.pos) for agent in self.agents]
            
        # You can still generate observations if needed for compatibility
        obs = state

        rewards = []
        for i, agent in enumerate(self.agents):
            
            reward = -1 #default penalty
            
            # reaching goal reward
            if self.is_agent_at_goal(i):
                reward = 10
            
            # lava penalty
            elif self.is_agent_in_lava(i):
                reward = -10
       
                
            rewards.append(reward)
            
        terminated = self.all_agents_at_goals()
        truncated = self.step_count >= self.max_steps

        info = {
            'step_count': self.step_count,
            'agent_positions': [tuple(agent.pos) for agent in self.agents],
            'collisions': collisions,
            'at_goals': [self.is_agent_at_goal(i) for i in range(len(self.agents))],
            'moves': moves
        }
        return obs, rewards, terminated, truncated, info
        
    def are_agents_colliding(self):
        """
        Check if agents are colliding (at the same position) or swapping positions.
        
        Returns:
            List of booleans indicating for each agent if it is colliding with any other agent
        """
        collisions = [False] * len(self.agents)
        
        # Track current positions
        current_positions = {}
        for i, agent in enumerate(self.agents):
            if agent.pos is None:
                continue
                
            pos_tuple = tuple(agent.pos)
            if pos_tuple in current_positions:
                # Mark current agent as colliding
                collisions[i] = True
                # Also mark the other agent that's at the same position
                collisions[current_positions[pos_tuple]] = True
            else:
                current_positions[pos_tuple] = i
        
        # Check for position swaps
        if hasattr(self, 'prev_positions'):
            for i, agent in enumerate(self.agents):
                if agent.pos is None or i not in self.prev_positions:
                    continue
                    
                current_pos = tuple(agent.pos)
                prev_pos = self.prev_positions[i]
                
                # Check if any other agent moved from current_pos to prev_pos (swap)
                for j, other_agent in enumerate(self.agents):
                    if i != j and j in self.prev_positions:
                        other_prev_pos = self.prev_positions[j]
                        other_current_pos = tuple(other_agent.pos) if other_agent.pos is not None else None
                        
                        if other_prev_pos == current_pos and other_current_pos == prev_pos:
                            collisions[i] = True
                            collisions[j] = True
        
        # Save current positions for next check
        self.prev_positions = {i: tuple(agent.pos) for i, agent in enumerate(self.agents) if agent.pos is not None}
        
        return collisions
    
    def all_agents_at_goals(self):
        """Check if all agents have reached their goals"""
        return all(self.is_agent_at_goal(i) for i in range(len(self.agents)))
    
    def visualize(self, highlight=False, highlight_masks=None, tile_size=TILE_PIXELS, show=True):
        """
        Visualize the environment using matplotlib
        
        Args:
            highlight (bool): Whether to highlight agent's view area
            highlight_masks: Optional custom mask to highlight specific cells
            tile_size (int): Size of each tile in pixels
            show (bool): Whether to display the image (True) or just return it (False)
        
        Returns:
            numpy.ndarray: The rendered grid image
        """
        # Si highlight_masks n'est pas fourni mais highlight est True,
        # générer un highlight_mask basé sur la visibilité des agents
        if highlight_masks is None and highlight:
            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = np.empty((self.width, self.height), dtype=object)
            for i in range(self.width):
                for j in range(self.height):
                    highlight_masks[i, j] = []

            for i, a in enumerate(self.agents):
                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = (
                    a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)
                )

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the whole grid
        grid_img = self.grid.render(
            tile_size=tile_size,
            highlight_masks=highlight_masks,
        )
        
        # Only display the image if show=True
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img)
            plt.axis('off')
            plt.show()
            
        return grid_img
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
                    actions.append(random.choice(list(CustomSmallActions)))  # Random action
            
            # Step the environment with the generated actions
            _, _, terminated, truncated, _ = self.step(actions)
            
            # Check if the episode is done
            if terminated or truncated:
                print("Episode ended")
                break
        
        # Final visualization
        self.visualize()
