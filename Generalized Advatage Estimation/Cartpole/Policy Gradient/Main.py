#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:51:37 2020

@author: yifanxu
"""
from Agent import *
import gym
import torch

env = gym.make('CartPole-v1')
SEED = 1234
env.seed(SEED)
#np.random.seed(SEED);
torch.manual_seed(SEED);

agent = Agent(4, 64, 2)
def main():
    
    
    for i in range(300):
        state = env.reset()
        eps_reward = 0
        for t in range(500):
            action = agent.select_action(torch.FloatTensor(state).unsqueeze(0))
            state, reward, done, _ = env.step(action.item())
            agent.rewards.append(reward)
            eps_reward += reward
            if(done):
                break
        print('iter', i, eps_reward)
        agent.update_policy()
    for i in range(10):
        state = env.reset()
        for t in range(3000):
            action = agent.select_action(torch.FloatTensor(state).unsqueeze(0))
            state, reward, done, _ = env.step(action.item())
            env.render()
if __name__ == '__main__':
    main()