#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:07:43 2020

@author: yifanxu
"""

from Agent import *
from Model import *
import gym
import torch

env = gym.make('Hopper-v2')
SEED = 1234
env.seed(SEED)
#np.random.seed(SEED);
torch.manual_seed(SEED);

actor = Action(11, 30, 3)
value = Value(11, 30)
agent = Agent(actor, value)
def main():
    
    average = 0
    for i in range(500):
        state = env.reset()
        eps_reward = 0
        for t in range(1000):
            action = agent.select_action(torch.FloatTensor(state).unsqueeze(0))
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            eps_reward += reward
            if(done):
                break
        average = average * 0.95 + eps_reward * 0.05
        agent.update_policy()
        #if(i % 30 == 0):
        print('iter', i, average)
    state = env.reset()
    for i in range (3000):
        action = agent.select_action(torch.FloatTensor(state).unsqueeze(0))
        state, reward, done, _ = env.step(action)
        env.render()
        
if __name__ == '__main__':
    main()