from gymnasium import spaces
import gymnasium as gym
import numpy as np
import random
import pickle

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
        
    def drop_nectar(self):
        nectar = self.nectar_collected
        self.nectar_collected = 0
        return nectar


class BeeHiveEnv(gym.Env):
    def __init__(self, grid_size=64, num_bees=2, view_size=1, max_nectar=1, max_steps = 100):
        # Initialize the grid size and number of bees
        self.grid_size = grid_size
        self.num_bees = num_bees
        self.max_steps = max_steps
        self.view_size= view_size
        self.max_nectar=max_nectar

        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, None actions for bees
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, grid_size, grid_size), dtype=np.int32)

        self.history = []
        self.episode = 0
        self.recording = True

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
                               x_padded - self.view_size:x_padded + self.view_size,
                               y_padded - self.view_size:y_padded + self.view_size]
    
        return bee_view
    
    def get_nearby_bees(self, target_bee):
        view_range = self.view_size
        nearby_bees = []
        #for other_bee in self.bees:
        #    if other_bee.bee_id == target_bee.bee_id:
        #        continue

        #    x_distance = abs(other_bee.x - target_bee.x)
        #    y_distance = abs(other_bee.y - target_bee.y)

        #    if x_distance <= view_range and y_distance <= view_range:
        #        nearby_bees.append(other_bee)
        
        for i in range(2 * self.view_size):
            for j in range(2 * self.view_size):
                nearby_bees.extend(self.grid_map.get((target_bee.x+i, target_bee.y+j), []))

        return nearby_bees
    
    
    
    def reset(self):
        self.grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int32)
        self.grid_map = dict()
        self.steps = 0
        
        center = self.grid_size // 2
        
        
        # Bottom layer (flowers)
        # Random grid with values 0 (no flower), 1 (flower with nectar), and -1 (flower without nectar). 
        num_flowers = [int(0.2 * self.grid_size ** 2), int(0.1 * self.grid_size ** 2)]
        flower_layer = np.zeros((self.grid_size, self.grid_size))
        xs = np.array(random.choices(list(range(self.grid_size)), k = sum(num_flowers)))
        ys = np.array(random.choices(list(range(self.grid_size)), k = sum(num_flowers)))
        flower_layer[xs[:num_flowers[0]], ys[:num_flowers[0]]] = 1
        flower_layer[xs[num_flowers[0]:], ys[num_flowers[0]:]] = -1
        #flower_layer = np.random.choice([0, 1, -1], size=(self.grid_size, self.grid_size), p=[0.7, 0.2, 0.1])
        self.flower_count = num_flowers[0]
        self.grid[0] = flower_layer

        if self.grid[0][center,center] != 0:
            self.grid[0][center,center] = 0
            

        # Place the hive in the center
        self.grid[1][center, center] = 1
        
        # Third layer (bees)
        self.bees = []
        for i in range(self.num_bees):
            x, y = np.random.randint(0, self.grid_size, size=2)
            self.bees.append(Bee(i, x, y, self.max_nectar))
            self.grid[2, x, y] = 1  # Bee layer
            self.grid_map[x,y] = self.grid_map.get((x, y), []) + self.bees[-1:]

        return [self.get_bee_observation(bee.x, bee.y) for bee in self.bees]

    def step(self, actions):
        """Each bee takes an action (list of actions, one per bee)."""
        reward_per_bee = np.array([0] * self.num_bees)
        self.steps += 1

        for i, bee in enumerate(self.bees):
            action = actions[i]
            
            # Check if another bee is there
            target_x, target_y = bee.x, bee.y
            if action == 0 and bee.x > 0:
                target_x -= 1
            elif action == 1 and bee.x < self.grid_size - 1:
                target_x += 1
            elif action == 2 and bee.y > 0:
                target_y -= 1
            elif action == 3 and bee.y < self.grid_size - 1:
                target_y += 1
            # If action == 4, we already have the target coordinates.
            
            
            self.grid_map[target_x, target_y] = self.grid_map.get((target_x, target_y), []) + [bee]
            self.grid_map[bee.x, bee.y].remove(bee)
            bee.x, bee.y = target_x, target_y

            # Collect nectar if standing on a flower
            if self.grid[0, bee.x, bee.y] == 1:
                if bee.collect_nectar():
                    reward_per_bee[i] += 1
                    self.grid[0, bee.x, bee.y] = -1  # Mark flower as empty
                    self.flower_count -= 1
            
            if self.grid[1, bee.x, bee.y] == 1:
                nectar = bee.drop_nectar()
                reward_per_bee[i] += nectar
        
        
        for loc, bees in self.grid_map.items():
            if bees:
                self.grid[2, loc[0], loc[1]] = 1
        
        obs = [self.get_bee_observation(bee.x, bee.y) for bee in self.bees]
        reward_per_bee = reward_per_bee - 0.1
        total_reward = np.sum(reward_per_bee)
        done = not np.any(self.grid[0] == 1) or self.steps > self.max_steps
        
        if self.recording: # Saves the bee level 0, flower level 1, and hive 2
            bofa = [self.grid_map, self.grid[0],self.grid[1]]
            self.history.append(bofa)
        if done:
            
            if self.recording:
                self.episode += 1
                with open(f'episode{str(self.episode)}.pkl', 'wb') as f:
                    pickle.dump(self.history, f)

                self.history = []

        return obs, reward_per_bee, total_reward, done, {}


    def save(self):
        print("\nFull Environment:")
        self.history.append(self.grid_map)
        #for layer in range(3):
            #print(f"Layer {layer} (0: Flowers, 1: Hive, 2: Bees)")
            #print(self.grid[layer])
            #print()

        #print("Bee Views:")
        #for bee in self.bees:
            #print(f"\nBee {bee.bee_id} at ({bee.x}, {bee.y}), Nectar: {bee.nectar_collected}/{bee.max_nectar} sees:")
            #bee_view = self.get_bee_observation(bee.x, bee.y)
            #for layer, layer_name in zip(range(3), ["Flowers", "Hive", "Bees"]):
                #print(f"{layer_name}:")
                #print(bee_view[layer])
            #print()
            
            
    def get_mask(self):
        mask = []
        for bee in self.bees:
            mask.append([0] * self.num_bees)
            nearby_bees = self.get_nearby_bees(bee)
            for other_bee in nearby_bees:
                mask[-1][other_bee.bee_id] = 1

        return np.array(mask)

    def close(self):
        pass

if __name__ == "__main__":
    env = BeeHiveEnv(grid_size=5, num_bees=2)
    state = env.reset()
    #env.render()

    for _ in range(1):
        actions = np.random.randint(0, 4, size=len(env.bees))
        state, reward, total_reward, done, _ = env.step(actions)
        env.save()
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}")
        if done:
            print("No nectar available. Episode ends.")
            break
