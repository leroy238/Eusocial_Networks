# Needs to import the environment and model. Placeholders for now.
import os
import random
import torch
import pickle
import random
from torch.optim import Adam
import numpy as np
from BeeModel.model import BeeNet_NoCom , BeeNet
from project_env import BeeHiveEnv as Environment
from torch.nn.utils.rnn import pad_sequence

VIEW_SIZE = 4
experience_buffer = []
# ext = "_3"

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
    batch, N, bees = rewards.shape
    N += 1
    # B x N + 1
    gammas = torch.tensor([gamma] * N, dtype = torch.float, device = rewards.device).pow(torch.arange(N, device = rewards.device))
    # N x B, 1 x B -> B x N + 1
    full_path = torch.cat((rewards, values.unsqueeze(1)), dim = 1).transpose(1,2)
    
    # B x N + 1, N + 1 -> B
    # print('full_path', full_path.shape)
    # print('gammgam', gammas.shape)
    return torch.bmm(full_path, gammas.unsqueeze(0).unsqueeze(-1).expand(batch,-1,1)).squeeze(-1)
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

    orig_lengths = torch.tensor([t.size(0)-1 for t in trajectory])   # (M,)
    traj = pad_sequence(trajectory, batch_first=False, padding_value=0).flatten(start_dim=1, end_dim=2)
    msk  = pad_sequence(mask, batch_first=False, padding_value=0).flatten(start_dim=1, end_dim=2)
    rew  = torch.stack(reward)
    act  = torch.stack(actions)
    # print('trajectory', traj.shape)
    # print('msk', msk.shape)
    # print('rew', rew.shape)
    # print('act', act.shape)

    with torch.no_grad():
        values = target(traj, msk, mini_batch = minibatch).unsqueeze(1).view(-1,minibatch,bees,5)[orig_lengths,torch.arange(minibatch)]
        values = torch.amax(values, dim = -1)
    
    y = n_step_TD(rew, values, gamma).flatten()
    Q = model(traj, msk, mini_batch = minibatch).unsqueeze(1).view(-1,minibatch,bees,5)[orig_lengths-1,torch.arange(minibatch)]
    Q = Q.view(-1,Q.shape[2])
    Q = Q[torch.arange(Q.shape[0]),act.flatten()]
    error = torch.sum(y - Q)
    loss = 1/(2 * minibatch * bees) * (error * error)
    # for i in range(minibatch):
    #     values = None
    #     with torch.no_grad():
    #         # M x T x B x C * L x C * L, M x T x B x comm ->  T x B * M x C * L x C * L, T x B * M x comm -> M * B x A
    #         # trajectory.view(trajectory.shape[1], -1, *(trajectory.shape[3:])), mask.view(mask.shape[1], -1, mask.shape[3])
    #         values = target(trajectory[i], mask[i])
    #         # print('trajectory', trajectory[i].shape)
    #         # print('mask', mask.shape)
    #         # print('values', values.shape)
    #         # M * B x A -> M x B x A
    #         #values = values.view(minibatch, -1, values.shape[2])
            
    #         # M x B x A -> M x B
    #         #dim = 2
    #         values = torch.amax(values, dim = 1)
    #     #end with
        
    #     # M x B
    #     y = n_step_TD(reward[i], values, gamma)
        
    #     # M x T - 1 x B x C * L x C * L, M x T - 1 x B x comm ->  T - 1 x B * M x C * L x C * L, T - 1 x B * M x comm -> M * B x A
    #     # trajectory.view(trajectory.shape[1] - 1, -1, *(trajectory.shape[3:])), communication[:, :-1].view(communication.shape[1] - 1, -1, communication.shape[3])
    #     Q = model(trajectory[i][:-1], mask[i][:-1])
    #     Q = Q[torch.arange(Q.shape[0]), actions[i]]
    #     # M * B x A -> M x B x A -> M x B
    #     #Q = Q.view(Q.shape[0], -1, Q.shape[2])[actions]
        
    #     # M x B, M x B -> 1
    #     error = torch.sum(y - Q)
    #     loss += 1/(2 * minibatch * bees) * (error * error)
    #end for
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
#end update_parameters

def save_model(model, reward , ext):
    model_path = os.path.join(os.getcwd(), "models", f"model{ext}.pth")
    reward_path = os.path.join(os.getcwd(), "rewards", f"reward{ext}.pkl")
    
    
    with open(model_path, "wb") as f:
        # pickle.dump(model, f)
        torch.save(model,model_path)
    #end with

    with open(reward_path, "wb") as f:
        pickle.dump(reward, f)
    #end with
#end save_model

def train(episodes, max_buffer, lr, gamma, epsilon, minibatch, target_update, num_bees,hidden_dim, N, decay, truncation, no_com=False):
    global experience_buffer
    env = Environment(num_bees=num_bees,view_size= VIEW_SIZE // 2, grid_size = 32, max_steps = 50)
    state = env.reset()
    Model = BeeNet_NoCom if no_com else BeeNet
    model = Model((num_bees,VIEW_SIZE,VIEW_SIZE),hidden_dim,env.action_space.n, truncation)
    target = Model((num_bees,VIEW_SIZE,VIEW_SIZE),hidden_dim,env.action_space.n, truncation)
    if torch.cuda.is_available():
        model = model.cuda()
        target = target.cuda()
    #end if
    target.load_state_dict(model.state_dict())
    tot_rewards = []

    
    optimizer = Adam(model.parameters(), lr = lr)
    steps = 0
    for i in range(episodes):
        state = env.reset()
        terminated = False
        rewards = []
        states = [state]
        actions = []
        masks = [env.get_mask()]
        epsilon_i = max(0.1, epsilon * decay ** (i // 10))

        
        while not(terminated):
            state_input = torch.tensor(np.array(states),dtype=torch.float)
            if torch.cuda.is_available():
                state_input = state_input.cuda()
            #end if
            with torch.no_grad():
                Q = model(state_input, torch.tensor(np.array(masks), device = state_input.device))[-1]
            r = random.random()
            a_t = torch.argmax(Q, axis = 1).squeeze() if r < epsilon_i else torch.randint(0, env.action_space.n, size = (num_bees,), device = Q.device)
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
            #end if/else
            steps += 1
            
            if steps % target_update == 0:
                target.load_state_dict(model.state_dict())
            #end if
        #end while

        tot_rewards.append(np.sum(np.array(rewards)))

        if i % 10 == 0:
            save_model(model, tot_rewards, f'eps_{episodes}_bffr_{max_buffer}_lr_{lr}_g_{gamma}_e_{epsilon}_mb_{minibatch}_tupdt_{target_update}_nbz_{num_bees}_hdim_{hidden_dim}_N_{N}_decay_{decay}_no_com_{no_com}')
        #end if

        print(f"Episode {i+1}/{episodes} - Total Reward: {tot_rewards[-1]:.2f} - Steps: {steps}", flush = True)
    #end for
#end train
