#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:14:52 2020

@author: yifanxu
"""
from Model import Actor
from Model import Critic
import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Normal
import numpy as np
import torch.optim as optim
from CriticTrainer import train_critic
from ActorTrainer import train_actor
from collections import deque
import gym
class Agent:
    def __init__(self, actor, critic, critic_lr=0.01):
        self.actor = actor
        self.critic = critic
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, state):
        '''
        This method takes the state of the environment as the input, and uses
        the gaussin distribution model that is modeled by the output(mean, std)
        of actor.forward() to sample the action
        input:
            state: a 1d list
        output:
            action: a 1d list
        '''
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, _, std = self.actor.forward(state)
        dist = torch.normal(mu, std)
        dist = dist.data
        return dist
    
    def update_policy(self, states, actions, rewards, masks):
        '''
        This method updates the policy accordingly to the memory using TRPO method
        input:
            states: 1d list of many dimensions
            actions: 1d list of n-d space list 
            rewards: 1d list
            masks: 1d list
        '''
        values = self.critic.forward(torch.FloatTensor(states))
       # train_critic(self.critic, self.critic_opt, states, rewards, masks)
        train_actor(self.actor, states, actions, values, rewards, masks)

actor = Actor(11, 50, 3)
critic = Critic(11, 50)

agent = Agent(actor, critic)

env = gym.make('Hopper-v2')
env.seed(500)
torch.manual_seed(500)
ava_reward = 0
for i in range(200):

    memory = deque()
    state = env.reset()
    rew = 0
    for m in range(1000):
        state = env.reset()
        eps_reward = 0
        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            eps_reward += reward
            if done:
                mask = 0
            else:
                mask = 1
            memory.append([state, action, reward, mask])
            state = next_state
            if done:
                break
        rew += eps_reward
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = memory[:, 1]
    actions = torch.cat([ action for action in actions])
    rewards = memory[:, 2]
    masks = memory[:, 3]
    agent.update_policy(states, actions, rewards, masks)
    #print(states)
    print('iter', i , rew / 1000)
        
for x in range (10):
    state = env.reset()
    rew = 0
    for t in range(100000):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        rew += reward
        if(done):
            break
        state = next_state
    print(rew)