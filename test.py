import sys
import json
import torch
import time
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
from REINFORCE import PolicyNetwork, update_policy


if __name__ == '__main__':

    sdf = prepare_indicators('data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')
    N = 500_000
    N2 = 700_000
    test_df = sdf[N:N2]
    
    test_env = BitcoinTradingEnv(test_df, lookback_window_size=60, 
                            commission=1e-4,  initial_balance=1000, serial=True)

    input_dim, seq_length = test_env.observation_space.shape
    output_dim1 = test_env.action_space.nvec[0]
    output_dim2 = test_env.action_space.nvec[1]
    hidden_dim = 128
    lstm_layers = 2

    # choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device used: {device}")

    policy_net = PolicyNetwork(input_dim, output_dim1, output_dim2, hidden_dim, n_layers=lstm_layers)
    # Loading the best model
    model_name = 'model/state_dict2.pt'
    policy_net.load_state_dict(torch.load(model_name))
    policy_net.to(device)

    policy_net.eval() # to tell the model how to treat dropout (train: uses dropout, eval: do not use dropout)
    state = test_env.reset()
    log_probs = []
    rewards = []
    profits = []
    hold_profits = []

    for steps in range(test_env.steps_left):
        if steps % 500 == 0:
            test_env.render()

        with torch.no_grad():
            action, log_prob = policy_net.get_action(state, device)
            new_state, reward, done, _ = test_env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            profits.append(test_env._get_profit())
            hold_profits.append(test_env._get_hold_profit())

            if done:
                print('\nTrading session terminated:')
                print(f"total reward: {np.round(np.sum(rewards), decimals=3)}")
                print(f"bot profit: {profits[-1]}")
                print(f"hold profit: {hold_profits[-1]}")
                print(f"steps: {steps}")
                break
            state = new_state

    # plot results
    fig, axs = plt.subplots(3)
    fig.suptitle('Trading results')
    axs[0].plot(profits, 'tab:blue')
    axs[0].set_title('Profit vs Time')
    axs[0].set(xlabel='Time (min)', ylabel='Profit ($)')
    axs[0].label_outer()
    axs[0].grid(True)

    axs[1].plot(hold_profits, 'tab:red')
    axs[1].set_title('Hold profit vs Time')
    axs[1].set(xlabel='Time (min)', ylabel='Hold profit ($)')
    axs[1].label_outer()
    axs[1].grid(True)

    axs[2].plot(rewards, 'tab:green')
    axs[2].set_title(f'Agent reward vs Time -- AuC = {np.sum(rewards)}')
    axs[2].set(xlabel='Time (min)', ylabel='Profit - Hold profit($)$')
    axs[2].label_outer()
    axs[2].grid(True)

    plt.savefig(f'model/test_fig_{N}_{N2}_{model_name}_{int(time.time())}.png')
    plt.show()