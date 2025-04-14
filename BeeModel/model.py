
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BeeNet(nn.Module):
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
        super(BeeNet, self).__init__()
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
    

