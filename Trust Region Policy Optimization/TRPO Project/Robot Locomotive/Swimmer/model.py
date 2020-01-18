#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:40:13 2020

@author: yifanxu
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, gamma=0.99, delta=0.01):
        super(Policy, self).__init__()
        self.gamma = gamma
        self.delta = delta
        self.l1 = nn.Linear(num_inputs, 30)
        self.l2 = nn.Linear(num_inputs, 30)

        self.action_head_mean = nn.Linear(30, 2)
        self.action_head_var = nn.Linear(30, 2)
       # self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.action_prb_records = []
        self.mu_records = []
        self.std_records = []
        self.rewards = []
        self.states = []

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        mu = F.tanh(self.action_head_mean(x1)) * 1
        var = F.sigmoid(self.action_head_var(x2)) * 0.1
        std = torch.sqrt(var)
        #print(mu)
        return mu, x, std
    
