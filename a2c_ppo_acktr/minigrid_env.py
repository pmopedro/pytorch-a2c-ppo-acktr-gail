from enum import IntEnum
from typing import Any, SupportsFloat
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces
import random
from gymnasium.core import ActType, ObsType
import numpy as np
import matplotlib.pyplot as plt


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    up = 2
    down = 3


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=13,
        fixed_layout=False,
        agent_start_pos=None,
        agent_start_dir=0,
        max_steps: int | None = None,
        replace_agent_episode=False,
        replace_goal_episode=False,
        seed=None,
        **kwargs,
    ):
        self.fixed_layout = fixed_layout
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.generated_maze = False
        self.replace_agent_each_episode = replace_agent_episode
        self.replace_goal_each_episode = replace_goal_episode
        self.goal_pos = None
        self.seed = seed  # Save the seed
        # Create a random state using the seed
        self.random_state = random.Random(seed)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 100

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Action enumeration for this environment
        self.actions = Actions

        # Define the observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(13, 13, 3),
                dtype='uint8'
            ),
            'direction': spaces.Discrete(4),
            # Adjust max_length as needed
            'mission': spaces.Text(max_length=100)
        })

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.reset()

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        # Initialize all interior cells as empty if the maze hasn't been generated yet
        if not self.generated_maze:
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

            # Clear all interior cells
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    self.grid.set(x, y, None)

            # Generate the maze by randomly placing walls
            self._generate_easy_maze(width, height)
            self.generated_maze = True
        else:
            # Clear previous agent and goal positions if they exist
            if self.agent_pos != (-1, -1):
                # Remove the agent from the grid
                self.grid.set(*self.agent_pos, None)

            # Remove the goal from the grid
            self.grid.set(*self.goal_pos, None)

        # Place the agent in a random valid position or use a predefined start position
        if self.replace_agent_each_episode or self.agent_pos == (-1, -1):
            if self.agent_start_pos is None:
                self._place_agent_randomly()
            else:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
                assert self.grid.get(
                    *self.agent_pos) is None, "Agent start position is not empty!"

        # Randomly place a goal square in a reachable position
        if self.replace_goal_each_episode or self.goal_pos is None:
            self._place_the_goal_randomly()
        else:
            # Re-place the goal at the same position
            self.put_obj(Goal(), *self.goal_pos)

        # Restore the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.mission = "grand mission"

    def _place_the_goal_randomly(self):
        empty_positions = [
            (x, y)
            for x in range(1, self.grid.width - 1)
            for y in range(1, self.grid.height - 1)
            if self.grid.get(x, y) is None
        ]

        if not empty_positions:
            raise ValueError(
                "No valid empty positions available to place the agent!")

        while True:
            goal_pos = self.random_state.choice(empty_positions)
            if self.grid.get(*goal_pos) is None:
                self.put_obj(Goal(), goal_pos[0], goal_pos[1])

                self.goal_pos = goal_pos  # Save the goal position
                break

    def _place_agent_randomly(self):
        empty_positions = [
            (x, y)
            for x in range(1, self.grid.width - 1)
            for y in range(1, self.grid.height - 1)
            if self.grid.get(x, y) is None
        ]

        if not empty_positions:
            raise ValueError(
                "No valid empty positions available to place the agent!")

        start_pos = self.random_state.choice(empty_positions)
        self.agent_pos = start_pos
        self.agent_dir = self.agent_start_dir

    def _generate_easy_maze(self, width, height):
        # Adjust wall probability to control difficulty
        wall_probability = 0.2  # Lower value for easier mazes

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if self.random_state.random() < wall_probability:
                    self.grid.set(x, y, Wall())
                else:
                    self.grid.set(x, y, None)

    def _is_reachable(self, start, goal):
        from collections import deque
        queue = deque([start])
        visited = set()
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if self.grid.get(nx, ny) is None and (nx, ny) not in visited:
                        queue.append((nx, ny))
                        visited.add((nx, ny))
        return False

    def plot_grid(self):
        img = self.gen_obs()

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the full grid observed by the agent.
        This method also outputs a visibility mask showing that all grid
        cells are fully visible.
        """
        # Use the full grid instead of a slice
        grid = self.grid  # Use the entire grid

        # Since it's fully observable, everything is visible
        vis_mask = np.ones(shape=(self.width, self.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        agent_pos = self.agent_pos
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (fully observable, high-resolution encoding)
        """

        # Get the fully observable grid and visibility mask
        grid, vis_mask = self.gen_obs_grid()

        # Encode the fully observable view into a numpy array
        image = grid.encode(vis_mask)

        image[self.agent_pos] = 2
        # Observations are dictionaries containing:
        # - an image (fully observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir,
               "mission": self.mission}

        return obs

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 * 0.99 ** (self.step_count)  # / self.max_steps

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position of the agent
        current_pos = self.agent_pos

        if action == self.actions.up:
            up_pos = tuple(a + b for a, b in zip(current_pos, (-1, 0)))
            up_cell = self.grid.get(*up_pos)
            terminated, reward = self.updata_step_custom(
                up_pos, up_cell, reward)

        elif action == self.actions.down:
            up_pos = tuple(a + b for a, b in zip(current_pos, (1, 0)))
            up_cell = self.grid.get(*up_pos)
            terminated, reward = self.updata_step_custom(
                up_pos, up_cell, reward)

        elif action == self.actions.right:
            up_pos = tuple(a + b for a, b in zip(current_pos, (0, 1)))
            up_cell = self.grid.get(*up_pos)
            terminated, reward = self.updata_step_custom(
                up_pos, up_cell, reward)

        elif action == self.actions.left:
            up_pos = tuple(a + b for a, b in zip(current_pos, (0, -1)))
            up_cell = self.grid.get(*up_pos)
            terminated, reward = self.updata_step_custom(
                up_pos, up_cell, reward)

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def updata_step_custom(self, next_pos, next_cell, reward):
        terminated = False
        if next_cell is None or next_cell.can_overlap():
            self.agent_pos = tuple(next_pos)
            if next_cell is not None and next_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if next_cell is not None and next_cell.type == "lava":
                terminated = True
        return terminated, reward
