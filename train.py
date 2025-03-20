# Needs to import the environment and model. Placeholders for now.
import os
import random
import torch
import numpy as np

class Environment:
    # step(action)
    # Input:
    #    action: Tuple<Tensor<B; B x D'>>
    # Output:
    #    Tuple<NdArray<B, C * L, C * L; B x B' x D'>>
    #    Float
    #    Boolean
    #
    # Placeholder for the environment with the intended output.
    def step(action):
        return (np.array([1]), np.array([1])), 0, False
    #end step
    
    def reset():
        pass
    #end reset
#end Environment

class Model:
    # __call__(x, comm, C_prev)
    # Input:
    #    x: Tensor<B, C * L, C * L>
    #    comm: Tensor<B, B', D'>
    #    C_prev: Tensor<B, D'>
    # Output:
    #    Tensor<B, A>
    #    Tensor<B, D'>
    # Takes as input the observation, the communication received by each bee,
    # and the communication given by each bee at the previous timestep. Returns the batched Q vector and communication.
    def __call__(self, x, comm, C_prev):
        return torch.tensor([1]), torch.tensor([1])
    #end __call__
    
    def cuda(self):
        return self
    #end cuda
#end Model

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
    K, B, N_back = reward.shape
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

def train(episodes, N, lr, gamma, K, C):
    env = Environment()
    state = env.reset()
    model = Model()
    target = Model()
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
        while not(terminated):
            C_prev = torch.zeros(states[0][1].shape)
            Q, comm = model(states[-1][0], states[-1][1], C_prev)
            a_t = (torch.argmax(Q, axis = 1).squeeze()
            actions.append(a_t)
            obs, reward, terminated = env.step(a_t.cpu().numpy(), comm)
            C_prev = states[-1][1]
            if len(states) < N:
                states.append(obs)
                rewards.append(reward)
            else:
                experience_buffer.append((states[0], torch.tensor(rewards, device = states[0].device), a_t, states[-1]))
                update_parameters(model, lr, gamma, K, optimizer)
                rewards = rewards[1:] + [reward]
                states = states[1:] + [obs]
            #end if/else
        #end while
        
        # After termination, add states leading up to termination.
        for j in range(1, len(states)-1):
            experience_buffer.append(states[j], torch.tensor(rewards[j:], device = states[j].device), actions[j], states[-1])
        #end for
        
        update_parameters(model, lr, gamma, K, optimizer)
        
        if i % C == 0:
            target.load_state_dict(model.state_dict())
        #end if
    #end for
#end train