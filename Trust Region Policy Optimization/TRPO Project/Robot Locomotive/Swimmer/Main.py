#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:25:44 2020

@author: yifanxu
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
from utils import *
from model import Policy
import math as m
import gym
from torch.distributions import Normal
from TrpoHelpers import *
from Agent import *
seed = 543
env = gym.make('Swimmer-v2')
env.seed(seed)
torch.manual_seed(seed)
gamma = 0.99

policy = Policy(8, 2)
agent = Agent(policy)


def main():
    for i in range(50):
        state = env.reset()
        eps_reward = 0
        for t in range(200):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            eps_reward += reward
            policy.rewards.append(reward)
            if(done):
                break
        print('episode ', i, eps_reward)
        agent.update_policy()
        agent.reset_policy_records()
    for t in range(1000):
        state = env.reset()
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        if(done):
            break

if __name__ == '__main__':
    main()