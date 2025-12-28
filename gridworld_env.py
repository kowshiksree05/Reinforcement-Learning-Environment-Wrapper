import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=5):
        super(GridWorldEnv, self).__init__()

        self.grid_size = grid_size

        # Actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation: (x, y)
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
        )

        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        self.goal = (4, 4)

        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        return np.array(self.agent_pos, dtype=np.float32)

    def step(self, action):
        x, y = self.agent_pos

        if action == 0:   # up
            x -= 1
        elif action == 1: # down
            x += 1
        elif action == 2: # left
            y -= 1
        elif action == 3: # right
            y += 1

        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)

        reward = -0.05  # small step penalty
        done = False

        if (x, y) in self.obstacles:
            reward = -1.0
        else:
            self.agent_pos = [x, y]

        if (x, y) == self.goal:
            reward = 10.0
            done = True

        return np.array(self.agent_pos, dtype=np.float32), reward, done, {}

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        for o in self.obstacles:
            grid[o] = "X"

        grid[self.goal] = "G"
        grid[tuple(self.agent_pos)] = "A"

        print(grid)