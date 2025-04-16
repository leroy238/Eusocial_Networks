
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class BeeNet(nn.Module):
    def __init__(self, inputdim, hidden_dim,action_space):
        super(BeeNet, self).__init__()
        # EYES
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(inputdim[1]*inputdim[2]*3,hidden_dim)
        self.relu = nn.ReLU()

        #Communication
        self.communication = torch.zeros((inputdim[0],hidden_dim))

        # LSTM
        self.lstm = nn.LSTMCell(input_size = hidden_dim*2 , hidden_size=hidden_dim)
        self.h_0 = nn.Parameter(torch.randn(inputdim[0], hidden_dim))
        self.c_0 = torch.zeros(inputdim[0], hidden_dim)
        # comm
        self.comnet = nn.Linear(hidden_dim , hidden_dim , bias=False)
        # q netowrk
        self.advatage = nn.Linear(hidden_dim , action_space)
        self.value = nn.Linear(hidden_dim , 1)

    
    def forward(self , states , comm_mask):
        #States should come in as shape <L, B , VIEW_SIZE,VIEW_SIZE>
        #Comm_mask should come in as shape <L, B , B , Hidden>

        # print(states.shape)
        states = states.view(states.shape[0],states.shape[1],states.shape[2]*states.shape[3]**2)
        
        eyes = self.relu(self.linear1(states))
        comm_mask = comm_mask.unsqueeze(-1).expand(-1,-1,-1,self.hidden_dim)
        print(comm_mask.shape)
        ht , ct = self.h_0 , self.c_0
        communication = self.communication.unsqueeze(1).expand(-1,self.communication.shape[0],-1) # <B,H> -> <B,B,H>
        for l in range(eyes.shape[0]):
            comm = communication * comm_mask[l]
            comm = torch.sum(torch.bmm(-torch.bmm(communication, comm.mT), comm), axis=-2) # <B,B,H> @ <B,H,B> @ <B,B,H> -> <B,H>

            input_t = torch.cat([eyes[l],comm],dim=-1) # -> <B,(2H)>
            # print(input_t.shape)

            ht , ct = self.lstm(input_t,(ht , ct))

            # comm network
            communication = self.comnet(ht).unsqueeze(1).expand(-1,self.communication.shape[0],-1)
            
            # qnet
            q = self.value(ht) + self.advatage(ht)









        return q


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
        self.advatage = nn.Linear(128 , action_space)
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
        
        q = self.value(ht) + self.advatage(ht)
        
        
        
        return q , comm_t , ht , ct
    

