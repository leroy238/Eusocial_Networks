import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Bee:
    def __init__(self, bee_id, x, y, max_nectar):
        self.bee_id = bee_id
        self.x = x
        self.y = y
        self.nectar_collected = 0
        self.max_nectar = max_nectar

    def collect_nectar(self):
        """Collects nectar if not at max capacity."""
        if self.nectar_collected < self.max_nectar:
            self.nectar_collected += 1
            return True
        return False


class BeeHiveEnv(gym.Env):
    def __init__(self, grid_size=5, num_bees=2, view_size=1, max_nectar=1):
        # Initialize the grid size and number of bees
        self.grid_size = grid_size
        self.num_bees = num_bees
        self.view_size= view_size
        self.max_nectar=max_nectar

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right actions for bees
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, grid_size, grid_size), dtype=np.int32)

        self.bees = []
        self.reset()

    
    def get_bee_observation(self, bee_x, bee_y):
        """
        Extracts the view around a bee, considering padding if near the edges.
        """
        pad_width = self.view_size
        padded_grid = np.pad(
            self.grid, 
            ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), 
            mode='constant', 
            constant_values=0  # Padding with 0
        )
    
        # Adjust coordinates because of padding
        x_padded = bee_x + pad_width
        y_padded = bee_y + pad_width
    
        # Extract the local area around the bee
        bee_view = padded_grid[:, 
                               x_padded - self.view_size:x_padded + self.view_size + 1,
                               y_padded - self.view_size:y_padded + self.view_size + 1]
    
        return bee_view
    
    
    
    def reset(self):
        self.grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int32)
        
        center = self.grid_size // 2
        
        
        # Bottom layer (flowers)
        # Random grid with values 0 (no flower), 1 (flower with nectar), and -1 (flower without nectar). 
        flower_layer = np.random.choice([0, 1, -1], size=(self.grid_size, self.grid_size), p=[0.7, 0.2, 0.1])
        self.grid[0] = flower_layer

        if self.grid[0][center,center] != 0:
            self.grid[0][center,center] = 0
            

        # Place the hive in the center
        self.grid[1][center, center] = 1  # Hive at the center
        
        # Third layer (bees)
        for i in range(self.num_bees):
            x, y = np.random.randint(0, self.grid_size, size=2)
            self.bees.append(Bee(i, x, y, self.max_nectar))
            self.grid[2, x, y] = 1  # Bee layer

        return [self.get_bee_observation(bee.x, bee.y) for bee in self.bees]

    def step(self, actions):
        """Each bee takes an action (list of actions, one per bee)."""
        nectar_found = 0

        for i, bee in enumerate(self.bees):
            action = actions[i]
            self.grid[2, bee.x, bee.y] = 0  # Clear old position

            if action == 0 and bee.x > 0:  # Up
                bee.x -= 1
            elif action == 1 and bee.x < self.grid_size - 1:  # Down
                bee.x += 1
            elif action == 2 and bee.y > 0:  # Left
                bee.y -= 1
            elif action == 3 and bee.y < self.grid_size - 1:  # Right
                bee.y += 1

            self.grid[2, bee.x, bee.y] = 1  # Update position

            # Collect nectar if standing on a flower
            if self.grid[0, bee.x, bee.y] == 1:
                if bee.collect_nectar():
                    nectar_found += 1
                    self.grid[0, bee.x, bee.y] = -1  # Mark flower as empty

        reward = nectar_found
        done = not np.any(self.grid[0] == 1)

        return [self.get_bee_observation(bee.x, bee.y) for bee in self.bees], reward, done, {}


    def render(self):
        print("\nFull Environment:")
        for layer in range(3):
            print(f"Layer {layer} (0: Flowers, 1: Hive, 2: Bees)")
            print(self.grid[layer])
            print()

        print("Bee Views:")
        for bee in self.bees:
            print(f"\nBee {bee.bee_id} at ({bee.x}, {bee.y}), Nectar: {bee.nectar_collected}/{bee.max_nectar} sees:")
            bee_view = self.get_bee_observation(bee.x, bee.y)
            for layer, layer_name in zip(range(3), ["Flowers", "Hive", "Bees"]):
                print(f"{layer_name}:")
                print(bee_view[layer])
            print()
            

    def close(self):
        pass

if __name__ == "__main__":
    env = BeeHiveEnv(grid_size=5, num_bees=2)
    state = env.reset()
    env.render()

    for _ in range(1):
        actions = np.random.randint(0, 4, size=len(env.bees))
        state, reward, done, _ = env.step(actions)
        env.render()
        print(f"Reward: {reward}")
        if done:
            print("No nectar available. Episode ends.")
            break
