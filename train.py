# Needs to import the environment and model. Placeholders for now.
import os
import random
import torch
from torch.optim import Adam
import numpy as np
from BeeModel.model import BeeNet as Model
from project_env import BeeHiveEnv as Environment

VIEW_SIZE = 4
experience_buffer = []

# n_step_TD(rewards, values, gamma)
# Input:
#    rewards: Tensor<B, N-1>
#    values: Tensor<B>
#    gamma: Float
# Output:
#    Tensor<B>
#
# Takes as input the rewards from N-1 steps, along with the value of the state after the N-th step, and the decay parameter. The input is batched by B.
# Returns a tensor that represents the TD value of the first state.
def n_step_TD(rewards, values, gamma):
    K, B, N_back = rewards.shape
    # K x B x N
    gammas = torch.tensor([gamma] * (N_back + 1)).pow(torch.arange(N_back + 1)).unsqueeze(0).unsqueeze(1).expand(K, B, -1)
    # K x B x N - 1, K x B x 1 -> K x B x N
    full_path = torch.cat((rewards, values.unsqueeze(2)), dim = 2)
    
    # K x B x N -> K x B
    return torch.bmm(gammas, full_path).sum(dim = 2)
#end n_step_TD

def update_parameters(model, target, lr, gamma, K, optimizer):
    mem_tuples = random.sample(experience_buffer, K)
    current, reward, actions, final = zip(*mem_tuples)
    
    current = torch.cat([c.unsqueeze(0) for c in current], dim = 0)
    reward = torch.cat([r.unsqueeze(0) for r in reward], dim = 0)
    actions = torch.cat([a.unsqueeze(0) for a in actions], dim = 0)
    final = torch.cat([f.unsqueeze(0) for f in final], dim = 0)
    
    values = None
    with torch.no_grad():
        # K * B x C * L x C * L -> K * B x A
        values, _ = target(final.view(-1, final.shape[2], final.shape[3]))
        # K * B x A -> K x B x A
        values = values.view(K, -1, values.shape[2])
        # K x B x A -> K x B
        values = torch.amax(values, dim = 2)
    #end with
    
    # K x B
    y = n_step_TD(reward, values)
    
    # K * B x C * L x C * L -> K * B x A
    Q = model(current.view(-1, current.shape[2], current.shape[3]))
    # K * B x A -> K x B x A -> K x B
    Q = Q.view(K, -1, Q.shape[2])[actions]
    
    loss = y - Q
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
#end update_parameters

def train(episodes, N, lr, gamma, K, C, B):
    env = Environment(num_bees=B,view_size= VIEW_SIZE / 2)
    state = env.reset()
    model = Model((B,VIEW_SIZE,VIEW_SIZE),env.action_space.n)
    target = Model((B,VIEW_SIZE,VIEW_SIZE),env.action_space.n)
    if torch.cuda.is_available():
        model = model.cuda()
        target = target.cuda()
    #end if
    target.load_state_dict(model.state_dict())

    
    optimizer = Adam(model.parameters(), lr = lr)
    for i in range(episodes):
        terminated = False
        rewards = []
        states = [state]
        actions = []
        comm = torch.zeros(B,128,dtype=torch.float)
        C_prev = comm

        while not(terminated):
            

            # # if there is a new error its definitley right here - Carson
            # combined_views = []

            # for bee in env.bees:
            #     bee_obs = torch.tensor(env.get_bee_observation(bee.x, bee.y))
            #     neighbor_obs_list = [bee_obs]

            #     nearby_bees = env.get_nearby_bees(bee)
            #     for other_bee in nearby_bees:
            #         other_obs = torch.tensor(env.get_bee_observation(other_bee.x, other_bee.y))
            #         neighbor_obs_list.append(other_obs)

            #     combined = torch.cat(neighbor_obs_list, dim=0)
            #     combined_views.append(combined)

            # comm_tensor = torch.stack(combined_views)


            # Q, comm = model(states[-1][0], states[-1][1], C_prev)
            # print("HERE: ", torch.tensor(states[-1],dtype=torch.float).shape)
            Q, comm = model(torch.tensor(states[-1],dtype=torch.float), comm, C_prev)
            a_t = torch.argmax(Q, axis = 1).squeeze()
            print(a_t)
            actions.append(a_t)
            obs, reward,total_reward ,terminated,_ = env.step(a_t.cpu().numpy())
            C_prev = comm
            if len(states) < N:
                states.append(obs)
                rewards.append(reward)
            else:
                experience_buffer.append((states[0], torch.tensor(rewards, device = states[0].device), a_t, states[-1]))
                update_parameters(model, target,lr, gamma, K, optimizer)
                rewards = rewards[1:] + [reward]
                states = states[1:] + [obs]
            #end if/else
        #end while
        
        # After termination, add states leading up to termination.
        for j in range(1, len(states)-1):
            experience_buffer.append(states[j], torch.tensor(rewards[j:], device = states[j].device), actions[j], states[-1])
        #end for
        
        update_parameters(model,target ,lr, gamma, K, optimizer)
        
        if i % C == 0:
            target.load_state_dict(model.state_dict())
        #end if

        
        print(f"Episode {i+1}/{episodes} - Total Reward: {total_reward:.2f}")
    #end for
#end train
