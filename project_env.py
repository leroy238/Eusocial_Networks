import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BeeHiveEnv(gym.Env):
    def __init__(self, grid_size=5, num_bees=2):
        # Initialize the grid size and number of bees
        self.grid_size = grid_size
        self.num_bees = num_bees
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right actions for bees
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, grid_size, grid_size), dtype=np.int32) # Three layers of size grid_size x grid_size which can have values between -1 and 1 inclusive
        self.reset()

    def reset(self):
        # Create the 3D grid with the dimensions (3, grid_size, grid_size)
        self.grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int32)
        
        # Bottom layer (flowers)
        # Random grid with values 0 (no flower), 1 (flower with nectar), and -1 (flower without nectar). 
        # The probabilities are set such that 70% are empty, 20% are flowers with nectar, and 10% are flowers without nectar.
        flower_layer = np.random.choice([0, 1, -1], size=(self.grid_size, self.grid_size), p=[0.7, 0.2, 0.1])
        self.grid[0] = flower_layer

        # Second layer (hive, always centered)
        center = self.grid_size // 2
        self.grid[1] = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[1][center, center] = 1  # Hive at the center
        
        # Third layer (bees)
        self.grid[2] = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for _ in range(self.num_bees):
            bee_x = np.random.randint(0, self.grid_size)
            bee_y = np.random.randint(0, self.grid_size)
            self.grid[2][bee_x, bee_y] = 1  # Place bees at random positions

        self.state = self.grid.copy()  # Save the state to return during each step
        return self.state

    def step(self, action):
        # Action: 0=Up, 1=Down, 2=Left, 3=Right
        bees_positions = np.argwhere(self.grid[2] == 1)

        # If there are no bees left on the grid (which shouldn't happen initially), 
        # the environment returns the state, zero reward, and a done flag set to True, indicating the episode is over.
        if len(bees_positions) == 0:
            return self.state, 0, True, {}

        # Move each bee based on the action
        for bee_pos in bees_positions:
            bee_x, bee_y = bee_pos
            if action == 0:  # Up
                if bee_x > 0:
                    self.grid[2][bee_x, bee_y] = 0
                    self.grid[2][bee_x-1, bee_y] = 1
            elif action == 1:  # Down
                if bee_x < self.grid_size - 1:
                    self.grid[2][bee_x, bee_y] = 0
                    self.grid[2][bee_x+1, bee_y] = 1
            elif action == 2:  # Left
                if bee_y > 0:
                    self.grid[2][bee_x, bee_y] = 0
                    self.grid[2][bee_x, bee_y-1] = 1
            elif action == 3:  # Right
                if bee_y < self.grid_size - 1:
                    self.grid[2][bee_x, bee_y] = 0
                    self.grid[2][bee_x, bee_y+1] = 1

        # Check if the bees are on flowers with nectar
        nectar_found = 0
        for bee_pos in np.argwhere(self.grid[2] == 1):
            bee_x, bee_y = bee_pos
            if self.grid[0][bee_x, bee_y] == 1:  # Flower with nectar
                nectar_found += 1
                self.grid[0][bee_x, bee_y] = -1  # Flower loses nectar

        reward = nectar_found  # Reward is the number of nectar flowers visited
         # Check if there is any nectar left in the environment
        nectar_left = np.any(self.grid[0] == 1)  # Check if there is any flower with nectar left
        done = not nectar_left  # Episode ends if no nectar is left anywhere
        
        self.state = self.grid.copy()
        return self.state, reward, done, {}

    def render(self):
        print("Flower Layer (0: no flower, 1: flower with nectar, -1: flower with no nectar):")
        print(self.grid[0])
        print("Hive Layer (1: hive at center):")
        print(self.grid[1])
        print("Bees Layer (1: bee present):")
        print(self.grid[2])

    def close(self):
        pass

if __name__ == "__main__":
    env = BeeHiveEnv(grid_size=5, num_bees=2)
    state = env.reset()
    env.render()

    # Example actions (up, down, left, right)
    for _ in range(10):
        action = np.random.randint(0, 4)  # Random action
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"Reward: {reward}")
        if done:
            print("No nectar available. Episode ends.")
            break
