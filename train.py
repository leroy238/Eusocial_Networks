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
    # N, B
    N, bees = rewards.shape
    N += 1
    # B x N + 1
    gammas = torch.tensor([gamma] * N, dtype = torch.float, device = rewards.device).pow(torch.arange(N, device = rewards.device))
    # N x B, 1 x B -> B x N + 1
    full_path = torch.cat((rewards, values.unsqueeze(0)), dim = 0).transpose(0,1)
    
    # B x N + 1, N + 1 -> B
    return full_path @ gammas
#end n_step_TD

def update_parameters(model, target, lr, gamma, minibatch, optimizer , bees):
    mem_tuples = random.sample(experience_buffer, minibatch)
    
    # zip takes an unrolled list of tuples, turns it into a tuple of lists
    trajectory, mask, reward, actions = zip(*mem_tuples)
    
    #trajectory = torch.nn.utils.rnn.pad_sequence(trajectory, batch_first = True, padding_value = -1.0)
    #mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first = True, padding_value = -1.0)
    #loss_mask = torch.all()
    
    # M x T x B x C * L x C * L
    #trajectory = torch.stack(padded_trajectory, dim = 0)
    # M x T x B x comm
    #mask = torch.stack(padded_mask, dim = 0)
    # M x T x B
    #reward = torch.stack(reward, dim = 0)
    # M x T x B
    #actions = torch.stack(actions, dim = 0)
    
    loss = 0
    for i in range(minibatch):
        values = None
        with torch.no_grad():
            # M x T x B x C * L x C * L, M x T x B x comm ->  T x B * M x C * L x C * L, T x B * M x comm -> M * B x A
            # trajectory.view(trajectory.shape[1], -1, *(trajectory.shape[3:])), mask.view(mask.shape[1], -1, mask.shape[3])
            values = target(trajectory[i], mask[i])
            # M * B x A -> M x B x A
            #values = values.view(minibatch, -1, values.shape[2])
            
            # M x B x A -> M x B
            #dim = 2
            values = torch.amax(values, dim = 1)
        #end with
        
        # M x B
        y = n_step_TD(reward[i], values, gamma)
        
        # M x T - 1 x B x C * L x C * L, M x T - 1 x B x comm ->  T - 1 x B * M x C * L x C * L, T - 1 x B * M x comm -> M * B x A
        # trajectory.view(trajectory.shape[1] - 1, -1, *(trajectory.shape[3:])), communication[:, :-1].view(communication.shape[1] - 1, -1, communication.shape[3])
        Q = model(trajectory[i][:-1], mask[i][:-1])
        Q = Q[torch.arange(Q.shape[0]), actions[i]]
        # M * B x A -> M x B x A -> M x B
        #Q = Q.view(Q.shape[0], -1, Q.shape[2])[actions]
        
        # M x B, M x B -> 1
        error = torch.sum(y - Q)
        loss += 1/(2 * minibatch * bees) * (error * error)
    #end for
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
#end update_parameters

def train(episodes, max_buffer, lr, gamma, minibatch, target_update, num_bees,hidden_dim, N):
    global experience_buffer
    env = Environment(num_bees=num_bees,view_size= VIEW_SIZE // 2)
    state = env.reset()
    model = Model((num_bees,VIEW_SIZE,VIEW_SIZE),hidden_dim,env.action_space.n)
    target = Model((num_bees,VIEW_SIZE,VIEW_SIZE),hidden_dim,env.action_space.n)
    if torch.cuda.is_available():
        model = model.cuda()
        target = target.cuda()
    #end if
    target.load_state_dict(model.state_dict())

    
    optimizer = Adam(model.parameters(), lr = lr)
    for i in range(episodes):
        state = env.reset()
        terminated = False
        rewards = []
        states = [state]
        actions = []
        masks = [env.get_mask()]

        steps = 0
        while not(terminated):
            state_input = torch.tensor(np.array(states),dtype=torch.float)
            if torch.cuda.is_available():
                state_input = state_input.cuda()
            #end if
            
            Q = model(state_input, torch.tensor(np.array(masks), device = state_input.device))
            a_t = torch.argmax(Q, axis = 1).squeeze()
            actions.append(a_t)
            
            obs, reward, total_reward, terminated, _ = env.step(a_t.cpu().numpy())
            mask = env.get_mask()
            
            masks.append(mask)
            states.append(obs)
            rewards.append(reward)
            
            if len(rewards) >= N:
                tup = (torch.tensor(np.array(states), dtype = torch.float, device = state_input.device), torch.tensor(np.array(masks), dtype = torch.float, device = state_input.device), torch.tensor(np.array(rewards[-N:]), dtype = torch.float, device = state_input.device), actions[-N])
                if len(experience_buffer) < max_buffer:
                    experience_buffer.append(tup)
                else:
                    experience_buffer = experience_buffer[1:] + [tup]
                #end if/else
            #end if
            
            if len(experience_buffer) > minibatch:
                update_parameters(model, target, lr, gamma, minibatch, optimizer, num_bees)
                steps += 1
            #end if/else
            
            if steps % target_update == 0:
                target.load_state_dict(model.state_dict())
            #end if
        #end while

        print(f"Episode {i+1}/{episodes} - Total Reward: {total_reward:.2f}")
    #end for
#end train
