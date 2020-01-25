#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:49:02 2020

@author: yifanxu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import math as m
discount_factor = 0.99
trace_decay = 0.99
LEARNING_RATE = 0.01

from GAE import *
from Model import *

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

class Agent:
    
    def __init__(self, actor, critic):
        
        self.policy = ActorCritic(actor, critic)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        
        self.values = []
        self.rewards = []
        #self.log_prob_actions = []
        self.mu = []
        self.std = []
        self.action_probs = []
    
    def select_action(self, state):
        mean, std, value = self.policy(state)
        #print(mean, std)
        dist = distributions.Normal(mean, std)
        action = dist.sample()
        self.mu.append(mean)
        self.std.append(std)
        #self.action_probs.append(action_prob)
        self.action_probs.append(action)
        self.values.append(value)
        return action
    
    def update_policy(self):
        rewards = self.rewards
        values = torch.cat(self.values).squeeze(-1)
        returns = calculate_returns(rewards, discount_factor)
        
        advantages = calculate_advantages(rewards, values, discount_factor, trace_decay)
        
        advantages = advantages.detach()
        returns = returns.detach()
        
        mu = torch.cat(self.mu)
        std = torch.cat(self.std)
        a = torch.cat(self.action_probs)
        self.optimizer.zero_grad()
        pi = 1/(2 * torch.pow(std, 2)) *  (torch.pow((a - mu), 2) / (std * std) -1)
        policy_loss = (pi.mean(dim=1) * advantages).sum()
        value_loss = F.smooth_l1_loss(returns, values).sum()
        self.optimizer.zero_grad()    
        policy_loss.backward()
        value_loss.backward()
        
        self.optimizer.step()
        self.values = []
        self.rewards = []
        #self.log_prob_actions = []
        self.mu = []
        self.std = []
        self.action_probs = []