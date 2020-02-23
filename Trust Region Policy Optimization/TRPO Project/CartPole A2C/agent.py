#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:12:19 2020

@author: yifanxu
"""

from  model import Actor
from model import Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hp
import torch.distributions as distributions
import gym
from collections import deque
import numpy as np
from trainer import train_actor
from trainer import train_critic
from GAE import calculate_returns
from GAE import calculate_advantages
class Agent:
    
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 0.01)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = distributions.Categorical(action_probs)
        action = dist.sample()
        #print(action_probs)
        return action
    
    def update_policy(self, states, actions, rewards, masks):
        states = torch.FloatTensor(states)
        values = self.critic(states).squeeze(-1)
        train_actor(self.actor, states, actions, rewards, masks, values)
        train_critic(self.critic_optimizer, states, actions, rewards, masks, values)

env = gym.make('CartPole-v1')
SEED = 1234
env.seed(SEED)
torch.manual_seed(SEED)

input_dim = env.observation_space.shape[0]
output_dim = 2
hidden_dim = hp.hidden_layer_size

actor = Actor(input_dim, hidden_dim, output_dim)
critic = Critic(input_dim, hidden_dim)

agent = Agent(actor, critic)

for i in range(1000):
    memory = deque()
    state = env.reset()
    eps_reward = 0
    for t in range(500):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        mask = 1 - done
        memory.append([state, action, reward, mask])
        state = next_state
        eps_reward += reward
        #print(next_state, action)
        if(done):
            break
    print('iter', i, eps_reward)
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = memory[:, 1]
    actions = torch.cat([ action for action in actions])
    rewards = memory[:, 2]
    masks = memory[:, 3]
    agent.update_policy(states, actions, rewards, masks)
    