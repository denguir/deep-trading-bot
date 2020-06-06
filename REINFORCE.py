import sys
import json
import torch
import gym
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.autograd import Variable
import matplotlib.pyplot as plt
import functools
from env.BitcoinTradingEnv import BitcoinTradingEnv
from env.indicators import prepare_indicators



# Constants
GAMMA = 0.9

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2, hidden_dim, n_layers=1, lr=3e-4, drop_prob=0.5):
        '''Recurrent neural network that is used as Policy for taking an action.
        - input_dim is the dimension of a input vector, i.e the number of indicators used
        - output_dim1 is the dimension of the first level of action: BUY, SELL, HOLD
        - output_dim2 is the dimension of the second level of action: 1/10, 2/10, ... of the portfolio
        '''  
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.hidden_dim = hidden_dim
        self.layer_dim = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.dropout = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(hidden_dim, output_dim1)
        self.linear2 = nn.Linear(hidden_dim + output_dim1, output_dim2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        lstm_out, _ = self.lstm(state.view(len(state), 1, -1))
        lstm_out = lstm_out.view(len(state), -1)[-1, :]
        lstm_out = self.dropout(lstm_out)
        temp_out = self.linear1(lstm_out)
    
        in2 = torch.cat((lstm_out, temp_out), dim=0)
        out1 = F.softmax(temp_out, dim=0)
        out2 = F.softmax(self.linear2(in2), dim=0)
        return out1, out2

    def get_action(self, state, device):
        state = torch.tensor(state.T, dtype=torch.float32).to(device) # transpose is easier for lstm to treat sequence of vectors
        probs1, probs2 = self.forward(Variable(state))
        probs1, probs2 = probs1.cpu(), probs2.cpu() # back to cpu
        highest_prob_action1 = np.random.choice(self.output_dim1, p=np.squeeze(probs1.detach().numpy())) 
        highest_prob_action2 = np.random.choice(self.output_dim2, p=np.squeeze(probs2.detach().numpy()))
        log_prob1 = torch.log(probs1[highest_prob_action1]) # log p(A)
        log_prob2 = torch.log(probs2[highest_prob_action2]) # log p(B|A)
        log_prob = log_prob1 + log_prob2 # log p(A and B) = log (p(A) * p(B|A)) = log P(A) + log P(B|A) 
        return (highest_prob_action1, highest_prob_action2), log_prob


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        # negative value because by default torch will do gradient descent, but we need to do gradient ascent
        policy_gradient.append(-log_prob * Gt) 

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()