#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:40:13 2020

@author: yifanxu
"""
import torch
import torch.autograd as autograd
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, gamma=0.99, delta=0.01):
        super(Policy, self).__init__()
        self.gamma = gamma
        self.delta = delta
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.action_prb_records = []
        self.mu_records = []
        self.std_records = []
        self.rewards = []
        self.states = []

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std) * 0.5

        return action_mean, action_log_std, action_std
    
