#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:27:21 2020

@author: yifanxu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions

from GAE import *
from Model import *

discount_factor = 0.99
trace_decay = 0.99
LEARNING_RATE = 0.01

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
class Agent:
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        #action = Policy(input_dim, hidden_dim, output_dim)
        #value = Value(input_dim, hidden_dim)
        actor = Policy(input_dim, hidden_dim, output_dim)
        critic = Value(input_dim, hidden_dim)
        self.policy = ActorCritic(actor, critic)
        #self.policy.apply(init_weights)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
    
    
    def select_action(self, state):
        action_prob, value_pred = self.policy(state)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        self.log_prob_actions.append(log_prob_action)
        self.values.append(value_pred)
        return action
    
    def update_policy(self):
        rewards = self.rewards
        values = torch.cat(self.values).squeeze(-1)
        log_prob_actions = torch.cat(self.log_prob_actions)
        returns = calculate_returns(rewards, discount_factor)
        
        advantages = calculate_advantages(rewards, values, discount_factor, trace_decay)
        #print(advantages)
        advantages = advantages.detach()
        returns = returns.detach()
        
        policy_loss = - (advantages * log_prob_actions).sum()
        value_loss = F.smooth_l1_loss(returns, values).sum()        
        self.optimizer.zero_grad()    
        policy_loss.backward()
        value_loss.backward()
        
        self.optimizer.step()
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        
        