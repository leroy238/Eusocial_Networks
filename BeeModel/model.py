
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class BeeNet(nn.Module):
    def __init__(self, inputdim, hidden_dim,action_space):
        super(BeeNet, self).__init__()
        # EYES
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(inputdim[1]*inputdim[2]*3 + 1,hidden_dim)
        self.relu = nn.ReLU()

        #Communication
        self.communication = torch.zeros((inputdim[0],hidden_dim))

        # LSTM
        self.lstm = nn.LSTMCell(input_size = hidden_dim*2 , hidden_size=hidden_dim)
        self.h_0 = nn.Parameter(torch.randn(inputdim[0], hidden_dim))
        # comm
        self.comnet = nn.Linear(hidden_dim , hidden_dim , bias=False)
        # q network
        self.advantage = nn.Linear(hidden_dim , action_space)
        self.value = nn.Linear(hidden_dim , 1)

    

    def forward(self , states , comm_mask , mini_batch = 1):
        #States should come in as shape <L, B , VIEW_SIZE,VIEW_SIZE>
        #Comm_mask should come in as shape <L, B , B , Hidden>
        q_l = []
        
        eyes = self.relu(self.linear1(states))
        comm_mask = comm_mask.unsqueeze(-1).expand(-1,-1,-1,self.hidden_dim)
        ht , ct = self.h_0.repeat(mini_batch,1) , torch.zeros((states.shape[1], self.hidden_dim), device = states.device)
        communication = torch.zeros((comm_mask.shape[1], comm_mask.shape[2], self.hidden_dim), device = states.device) # <B,H> -> <B,B,H>
        
        for l in range(eyes.shape[0]):

            # print('communication',communication.shape)
            # print('comm_mask',comm_mask[l].shape)
            comm = communication * comm_mask[l]
            comm = torch.sum(torch.bmm(-torch.bmm(communication, comm.mT), comm), axis=-2) # <B,B,H> @ <B,H,B> @ <B,B,H> -> <B,H>

            input_t = torch.cat([eyes[l],comm],dim=-1) # -> <B,(2H)>
            # print(input_t.shape)

            ht , ct = self.lstm(input_t,(ht , ct))

            # comm network
            communication = self.comnet(ht).unsqueeze(1).expand(-1,self.communication.shape[0],-1)
            
            # qnet
            A = self.advantage(ht)
            q = self.value(ht) + A - torch.mean(A,dim=-1,keepdim=True)
            q_l.append(q)









        return torch.stack(q_l)
    
class BeeNet_NoCom(nn.Module):
    def __init__(self, inputdim, hidden_dim,action_space):
        super(BeeNet_NoCom, self).__init__()
        # EYES
        self.hidden_dim = hidden_dim * 2

        self.linear1 = nn.Linear(inputdim[1]*inputdim[2]*3 + 1,hidden_dim*2)
        self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTMCell(input_size = hidden_dim*2 , hidden_size=hidden_dim*2)
        self.h_0 = nn.Parameter(torch.randn(inputdim[0], hidden_dim*2))
        # q netowrk
        self.advantage = nn.Linear(hidden_dim * 2 , action_space)
        self.value = nn.Linear(hidden_dim * 2, 1)

    

    def forward(self , states , comm_mask , mini_batch = 1):
        #States should come in as shape <L, B , VIEW_SIZE,VIEW_SIZE>
        #Comm_mask should come in as shape <L, B , B , Hidden>
        q_l = []
        
        eyes = self.relu(self.linear1(states))
        comm_mask = comm_mask.unsqueeze(-1).expand(-1,-1,-1,self.hidden_dim)
        ht , ct = self.h_0.repeat(mini_batch,1) , torch.zeros((states.shape[1], self.hidden_dim), device = states.device)
        
        for l in range(eyes.shape[0]):
            ht , ct = self.lstm(eyes[l],(ht , ct))
            
            # qnet
            q = self.value(ht) + self.advantage(ht)
            q_l.append(q)

        return torch.stack(q_l)


class BeeNet2(nn.Module):
    # Input:
    #    x: Tensor<B, C * L, C * L>
    #    comm: Tensor<B, B', D'>
    #    C_prev: Tensor<B, D'>
    # Output:
    #    Tensor<B, A>
    #    Tensor<B, D'>
    # Takes as input the observation, the communication received by each bee,
    # and the communication given by each bee at the previous timestep. Returns the batched Q vector and communication.
    
    
    def __init__(self, inputdim , action_space):
        super(BeeNet2, self).__init__()
        # self.eyes = self.Eyes(inputdim = inputdim ,render_style='bitmap')
        
        
        # EYES
        self.linear1 = nn.Linear(inputdim[1]*inputdim[2]*3,128)
        self.relu = nn.ReLU()
        
        # LSTM
        
        self.lstm = nn.LSTMCell(input_size = 128 , hidden_size=128)
        self.h_0 = nn.Parameter(torch.randn(inputdim[0], 128))
        self.c_0 = torch.zeros(inputdim[0], 128)
        self.h_t = None
        self.c_t = None
        self.start = True
        
        # comm
        self.comnet = nn.Linear(128 , 128 , bias=False)
        
        # q netowrk
        self.advantage = nn.Linear(128 , action_space)
        self.value = nn.Linear(128 , 1)
    
    def forward(self, state , comm , C_prev):
        
        # EYES
        
        input_t = self.relu(self.linear1(state.flatten(start_dim = 1))) #Tensor<B, C * L, C * L> -> Tensor<B, 128>
        
        
        comms = torch.sum(-(comm.T @ C_prev) @ comm, axis=-1)
        
        input_t = input_t + comms

        print(input_t.shape)
        print(comms.shape)
        
        #LSTM
        if self.start:
            ht , ct = self.lstm(input_t,(self.h_0,self.c_0))
        else:
            ht , ct = self.lstm(input_t,(self.h_t,self.c_t))
        
        self.h_t , self.c_t = ht , ct
        
        # comm network
        comm_t = self.comnet(ht)
        
        # qnet
        
        q = self.value(ht) + self.advantage(ht)
        
        
        
        return q , comm_t , ht , ct
    

