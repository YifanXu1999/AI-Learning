#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:20:07 2020

@author: yifanxu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = F.relu(self.drop(self.l1(x)))
        #x = F.relu(self.l1(x))
        x = self.l2(x)
        return F.softmax(x, dim=-1)
    
class Value(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = F.relu(self.drop(self.l1(x)))
        #x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
    
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred