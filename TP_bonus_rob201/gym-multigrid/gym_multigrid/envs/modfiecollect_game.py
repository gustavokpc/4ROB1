import random
import matplotlib.pyplot as plt
import enum
import numpy as np

from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.agent import Agent
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Wall, Ball, Lava
from gym_multigrid.core.world import World
from gym_multigrid.core.constants import COLORS

class BallCollectActions(enum.IntEnum):
    stay = 0
    left = 1
    right = 2
    up = 3
    down = 4

    def available(self):
        return ["move"]

class EnterableLava(Lava):
    def can_enter(self):
        return True

class BallCollectEnv(MultiGridEnv):
    """
    Multi-agent Ball Collection environment (MultiGrid style).
    Each agent must collect its own balls.
    Internal walls can be replaced by enterable lava (penalty if entered).
    """

    def __init__(
        self,
        width=8,
        height=8,
        max_steps=100,
        agents_positions=None,
        balls_by_agent=None,
        seed=None
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.step_count = 0

        if agents_positions is None:
            agents_positions = [(1, 1)]
        self.agents_positions = agents_positions
        self.num_agents = len(agents_positions)

        # Gestion flexible du format balls_by_agent pour train/test, 1 ou plusieurs agents
        if isinstance(balls_by_agent, list):
            if self.num_agents == 1:
                # Autorise [(x, y), (x2, y2)] ou [[(x, y), (x2, y2)]]
                if len(balls_by_agent) == 0 or isinstance(balls_by_agent[0], tuple):
                    self.balls_by_agent = [balls_by_agent]
                elif isinstance(balls_by_agent[0], list):
                    self.balls_by_agent = balls_by_agent
                else:
                    raise ValueError("balls_by_agent mal formé pour 1 agent.")
            else:
                # Plusieurs agents : doit être liste de listes
                if all(isinstance(b, list) for b in balls_by_agent):
                    self.balls_by_agent = balls_by_agent
                else:
                    raise ValueError("En mode test multi-agent, balls_by_agent doit être une liste de listes de tuples.")
        else:
            raise ValueError("balls_by_agent doit être une liste.")
            
        OBJECT_TO_IDX = {
            "unseen": 0,
            "empty": 1,
            "wall": 2,
            "floor": 3,
            "goal": 8,
            "ball": 11,
            "agent": 10,
            "lava": 9,
        }
        self.world = World(
            encode_dim=3,
            normalize_obs=False,
            OBJECT_TO_IDX=OBJECT_TO_IDX,
            COLORS=COLORS
        )
        agents_list = [Agent(self.world) for _ in range(self.num_agents)]

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            agents=agents_list,
            partial_obs=False,
            agent_view_size=7,
            actions_set=BallCollectActions,
            world=self.world,
            render_mode="rgb_array",
            uncached_object_types=[]
        )

        if seed is not None:
            self.seed(seed)

        self.collect_reward = 100
        self.wrong_collect_penalty = -100
        self.collision_penalty = -10
        self.step_penalty = -1.0

        self._gen_grid(self.width, self.height)

    def _gen_grid(self, width, height, world=None):
        self.grid = Grid(width, height, world=self.world)
        self.grid.wall_rect(0, 0, width, height)

        # Remplacer uniquement les murs internes par de la lave "enterable"
        self.lava_positions = set()
        for x in range(1, width-1):
            for y in range(1, height-1):
                cell = self.grid.get(x, y)
                if isinstance(cell, Wall):
                    self.grid.set(x, y, EnterableLava(self.world))
                    self.lava_positions.add((x, y))

        agent_colors = ['red', 'blue', 'green', 'purple', 'yellow', 'orange']
        self.agent_balls = {}
        for i, pos in enumerate(self.agents_positions):
            color = agent_colors[i % len(agent_colors)]
            agent = self.agents[i]
            agent.color = color
            self.place_agent(agent, pos)
            self.agent_balls[i] = set(self.balls_by_agent[i])

        for agent_id, balls in enumerate(self.balls_by_agent):
            for ball_pos in balls:
                assert isinstance(ball_pos, tuple) and len(ball_pos) == 2, f"ball_pos mal formé: {ball_pos}"
                ball = Ball(self.world)
                ball.color = agent_colors[agent_id % len(agent_colors)]
                ball.owner_id = agent_id
                self.grid.set(*ball_pos, ball)

        self.balls_collected = {i: 0 for i in range(self.num_agents)}
        self.balls_collected_wrong = {i: 0 for i in range(self.num_agents)}
        self.step_count = 0

    def add_wall(self, x, y):
        """
        Ajoute une cellule de lave "enterable" à la position (x, y) sur la grille.
        """
        # Ne pas autoriser sur les bords
        if x == 0 or y == 0 or x == self.width-1 or y == self.height-1:
            return
        self.grid.set(x, y, EnterableLava(self.world))
        if not hasattr(self, "lava_positions"):
            self.lava_positions = set()
        self.lava_positions.add((x, y))

    def get_lava_pos(self):
        """
        Retourne la liste des positions de lave (EnterableLava) dans l'environnement.
        """
        return list(self.lava_positions)

    def step(self, actions):
        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions]
        if len(actions) != self.num_agents:
            actions = actions + [0] * (self.num_agents - len(actions))

        rewards = [self.step_penalty for _ in range(self.num_agents)]

        # Sauvegarder les anciennes positions des agents
        old_positions = [tuple(agent.pos) for agent in self.agents]

        # Déplacer chaque agent selon son action
        for agent_id, action in enumerate(actions):
            self._move_agent(agent_id, action)

        # Retirer les agents à leur ancienne position
        for pos in old_positions:
            self.grid.set(*pos, None)

        # Vérifier la collecte de balle par position
        for agent_id, agent in enumerate(self.agents):
            pos = tuple(agent.pos)
            if pos in self.agent_balls[agent_id]:
                rewards[agent_id] = self.collect_reward
                self.balls_collected[agent_id] += 1
                self.agent_balls[agent_id].remove(pos)
                cell = self.grid.get(*pos)
                if isinstance(cell, Ball):
                    self.grid.set(*pos, None)
            else:
                for other_id in range(self.num_agents):
                    position_other_agent = tuple(self.agents[other_id].pos)
                    if other_id != agent_id and pos in self.agent_balls[other_id]:
                        rewards[agent_id] = self.wrong_collect_penalty
                        self.balls_collected_wrong[agent_id] += 1
                        self.agent_balls[other_id].remove(pos)
                        cell = self.grid.get(*pos)
                        if isinstance(cell, Ball):
                            self.grid.set(*pos, None)
                        break
                    elif pos == position_other_agent and other_id != agent_id:
                        rewards[agent_id] += self.collision_penalty
                        
                
        # Ajout de la pénalité pour la lave
        for agent_id, agent in enumerate(self.agents):
            pos = tuple(agent.pos)
            cell = self.grid.get(*pos)
            if isinstance(cell, EnterableLava):
                rewards[agent_id] = -50.0

        # Enfin, on place tous les agents sur la grille
        for agent in self.agents:
            self.grid.set(*agent.pos, agent)

        self.step_count += 1

        if self.num_agents == 1:
            terminated = self._all_balls_collected(agent_id=0)
        else:
            terminated = self._all_balls_collected()
        truncated = self.step_count >= self.max_steps

        obs = [tuple(agent.pos) for agent in self.agents]
        info = self._get_info()
        return obs, rewards, terminated, truncated, info

    def _move_agent(self, agent_id, action):
        agent = self.agents[agent_id]
        pos = list(agent.pos)
        new_pos = pos.copy()
        # 0=stay, 1=left, 2=right, 3=up, 4=down
        if action == 1:
            
            new_pos[0] -= 1
        elif action == 2:
            new_pos[0] += 1
        elif action == 3:
            new_pos[1] -= 1
        elif action == 4:
            new_pos[1] += 1

        # Interdire de sortir de la grille
        if not (0 < new_pos[0] < self.width-1 and 0 < new_pos[1] < self.height-1):
            return
        cell = self.grid.get(*new_pos)
        if isinstance(cell, Wall):
            return
        # EnterableLava: agent peut entrer
        agent.pos = new_pos

    def _all_balls_collected(self, agent_id=None):
        if agent_id is not None:
            return len(self.agent_balls.get(agent_id, [])) == 0
        for balls in self.agent_balls.values():
            if balls:
                return False
        return True

    def _get_info(self):
        return {
            'balls_collected': self.balls_collected.copy(),
            'balls_collected_wrong': self.balls_collected_wrong.copy(),
        }

    def visualize(self, highlight=False, highlight_masks=None, tile_size=32, show=True):
        if highlight_masks is None and highlight:
            _, vis_masks = self.gen_obs_grid()
            highlight_masks = np.empty((self.width, self.height), dtype=object)
            for i in range(self.width):
                for j in range(self.height):
                    highlight_masks[i, j] = []
            for i, a in enumerate(self.agents):
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = (
                    a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)
                )
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        if not vis_masks[i][vis_i, vis_j]:
                            continue
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue
                        highlight_masks[abs_i, abs_j].append(i)
        grid_img = self.grid.render(
            tile_size=tile_size,
            highlight_masks=highlight_masks,
        )
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img)
            plt.axis('off')
            plt.show()
        return grid_img

    def run_random_agents(self, max_steps=100):
        self.reset()
        self.visualize(show=True)
        for step in range(max_steps):
            actions = [random.randint(0, 4) for _ in range(self.num_agents)]
            obs, rewards, terminated, truncated, info = self.step(actions)
            print(f"Step {step+1}: Actions={actions}, Rewards={rewards}")
            self.visualize(show=True)
            if terminated or truncated:
                print(f"Episode terminé à l'étape {step+1}")
                break
        print("Statistiques finales:")
        print(f"Balles correctement collectées: {self.balls_collected}")
        print(f"Balles incorrectement collectées: {self.balls_collected_wrong}")

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._gen_grid(self.width, self.height)
        obs = [tuple(agent.pos) for agent in self.agents]
        return obs, self._get_info()
        