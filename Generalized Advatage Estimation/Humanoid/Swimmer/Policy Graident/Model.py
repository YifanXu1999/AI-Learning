#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:39:30 2020

@author: yifanxu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Action(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        #self.action_mean.weight.data.mul_(0.1)
        #self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, output_dim))
    
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_std
    
class Value(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        #self.output.weight.data.mul_(0.1)
        #self.output.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.output(x))
        
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        action_mean, action_std = self.actor(state)
        value = self.critic(state)
        
        return action_mean, action_std, value