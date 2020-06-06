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
    train_df = sdf[:N]

    train_env = BitcoinTradingEnv(train_df, lookback_window_size=60, 
                            commission=1e-4, initial_balance=1000, serial=False)
    
    input_dim, seq_length = train_env.observation_space.shape
    output_dim1 = train_env.action_space.nvec[0]
    output_dim2 = train_env.action_space.nvec[1]
    hidden_dim = 128
    lstm_layers = 2

    # choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    print(f"Device used: {device}")

    policy_net = PolicyNetwork(input_dim, output_dim1, output_dim2, hidden_dim, n_layers=lstm_layers)
    # Loading the best model
    model_name = 'model/state_dict3.pt'
    policy_net.load_state_dict(torch.load(model_name))
    policy_net.to(device)

    max_episode_num = 1_000
    all_rewards = [0]
    avg_rewards = [0]

    policy_net.train() # to tell the model how to treat dropout (train: uses dropout, eval: do not use dropout)
    for episode in range(max_episode_num):
        state = train_env.reset()
        log_probs = []
        rewards = []
        profits = []
        hold_profits = []

        for steps in range(train_env.steps_left):
            if steps % 500 ==0:
                train_env.render()

            action, log_prob = policy_net.get_action(state, device)
            new_state, reward, done, _ = train_env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            profits.append(train_env._get_profit())
            hold_profits.append(train_env._get_hold_profit())

            if done:
                # Evaluate model
                all_rewards.append(np.mean(rewards))
                avg_rewards.append(np.mean(all_rewards[-10:]))

                if all_rewards[-1] > all_rewards[-2]:
                    torch.save(policy_net.state_dict(), 'model/state_dict3.pt')
                    print(f'Reward increased ({episode - 1} --> {episode}).\nSaving model ...')

                update_policy(policy_net, rewards, log_probs)

                if (episode % 1) == 0:
                    print(f'\nDone episode {episode}')
                    print(f"total reward: {np.round(np.sum(rewards), decimals=3)}")
                    print(f"bot profit: {profits[-1]}")
                    print(f"hold profit: {hold_profits[-1]}")
                    print(f"steps: {steps}")
                break
            
            state = new_state

    # plot results
    plt.plot(all_rewards)
    plt.plot(avg_rewards)
    plt.plot([0] * len(all_rewards))
    plt.grid(True)
    plt.title(f'AuC = {np.sum(all_rewards)}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'model/train_fig_{int(time.time())}.png')
    plt.show()