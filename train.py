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
    # M, B, N
    minibatch, bees, N = rewards.shape
    N += 1
    # M x B x N
    gammas = torch.tensor([gamma] * N, device = rewards.device).pow(torch.arange(N)).unsqueeze(0).unsqueeze(1).expand(minibatch, bees, -1)
    # M x B x N - 1, M x B x 1 -> M x B x N
    full_path = torch.cat((rewards, values.unsqueeze(2)), dim = 2)
    
    # M x B x N -> M x B
    return torch.bmm(gammas, full_path).sum(dim = 2)
#end n_step_TD

def update_parameters(model, target, lr, gamma, minibatch, optimizer):
    mem_tuples = random.sample(experience_buffer, minibatch)
    
    # zip takes an unrolled list of tuples, turns it into a tuple of lists
    trajectory, communication, reward, actions = zip(*mem_tuples)
    
    # M x T x B x C * L x C * L
    trajectory = torch.stack(trajectory, dim = 0)
    # M x T x B x comm
    communication = torch.stack(communication, dim = 0)
    # M x T x B
    reward = torch.stack(reward, dim = 0)
    # M x T x B
    actions = torch.stack(actions, dim = 0)
    
    values = None
    with torch.no_grad():
        # M x T x B x C * L x C * L, M x T x B x comm ->  T x B * M x C * L x C * L, T x B * M x comm -> M * B x A
        values, _ = target(trajectory.view(trajectory.shape[1], -1, *(trajectory.shape[3:])), communication.view(communication.shape[1], -1, communication.shape[3]))
        # M * B x A -> M x B x A
        values = values.view(minibatch, -1, values.shape[2])
        
        # M x B x A -> M x B
        values = torch.amax(values, dim = 2)
    #end with
    
    # M x B
    y = n_step_TD(reward, values)
    
    # M x T - 1 x B x C * L x C * L, M x T - 1 x B x comm ->  T - 1 x B * M x C * L x C * L, T - 1 x B * M x comm -> M * B x A
    Q = model(trajectory[:, :-1].view(trajectory.shape[1] - 1, -1, *(trajectory.shape[3:])), communication[:, :-1].view(communication.shape[1] - 1, -1, communication.shape[3]))
    # M * B x A -> M x B x A -> M x B
    Q = Q.view(K, -1, Q.shape[2])[actions]
    
    # M x B, M x B -> 1
    error = torch.sum(y - Q)
    loss = 1/(2 * minibatch * bees) * (error * error)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
#end update_parameters

def train(episodes, max_buffer, lr, gamma, minibatch, target_update, num_bees, N):
    env = Environment(num_bees=num_bees,view_size= VIEW_SIZE // 2)
    state = env.reset()
    model = Model((num_bees,VIEW_SIZE,VIEW_SIZE),env.action_space.n)
    target = Model((num_bees,VIEW_SIZE,VIEW_SIZE),env.action_space.n)
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
        comm = torch.zeros(num_bees,128,dtype=torch.float)
        if torch.cuda.is_available():
            comm = comm.cuda()
        #end if
        comms = [comm]

        steps = 0
        while not(terminated):
            # combined_views = []

            # Maybe consider moving this into the environment since it uses variables that should be private - Justin
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


            state_input = torch.tensor(states,dtype=torch.float, device = comm.device)
            comm_time
            Q = model(state_input, masks)
            a_t = torch.argmax(Q, axis = 1).squeeze()
            actions.append(a_t)
            
            obs, reward, total_reward, terminated, _ = env.step(a_t.cpu().numpy())
            mask = env.get_mask()
            
            comms.extend(comm)
            states.append(obs)
            rewards.append(reward)
            
            if len(states) >= N:
                if len(experience_buffer) < max_buffer:
                    experience_buffer.append((states, torch.stack(comms), torch.tensor(rewards[-N:], states[0].device), actions[-N]))
                else:
                    experience_buffer = experience_buffer[1:] + (states, torch.tensor(rewards[-N:], states[0].device), actions[-N])
                #end if/else
            #end if
            
            if len(experience_buffer) > minibatch:
                update_parameters(model, target, lr, gamma, minibatch, optimizer)
                steps += 1
            #end if/else
            
            if steps % C == 0:
                target.load_state_dict(model.state_dict())
            #end if
        #end while

        print(f"Episode {i+1}/{episodes} - Total Reward: {total_reward:.2f}")
    #end for
#end train
