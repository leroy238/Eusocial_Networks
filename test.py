import os
import random
import torch
import pickle
import random
import numpy as np
from BeeModel.model import BeeNet as Model
from project_env import BeeHiveEnv as Environment

VIEW_SIZE = 4

def load_model():
    with open(os.path.join(os.getcwd(), "models", "model_Com_adam.pkl"), "rb") as f:
        model = pickle.load(f)
    #end with
    
    return model
#end load_model

def test(num_bees, hidden_dim):
    env = Environment(num_bees=num_bees,view_size= VIEW_SIZE // 2, grid_size = 32, max_steps = 50)
    state = env.reset()
    model = load_model()#Model((num_bees,VIEW_SIZE,VIEW_SIZE),hidden_dim,env.action_space.n)
    if torch.cuda.is_available():
        model = model.cuda()
    #end if
    
    state = env.reset()
    terminated = False
    rewards = []
    states = [state]
    masks = [env.get_mask()]

    steps = 0
    while not(terminated):
        state_input = torch.tensor(np.array(states),dtype=torch.float)
        if torch.cuda.is_available():
            state_input = state_input.cuda()
        #end if
        
        Q = model(state_input, torch.tensor(np.array(masks), device = state_input.device))[-1]
        a_t = torch.argmax(Q, axis = 1).squeeze()
        
        obs, reward, total_reward, terminated, _ = env.step(a_t.cpu().numpy())
        mask = env.get_mask()
        
        masks.append(mask)
        states.append(obs)
        rewards.append(total_reward)
    #end while
    print(rewards)
#end test

test(32, 128)
