 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:04:07 2020

@author: yifanxu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_dim,  hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        #self.log_std_layer = nn.Linear(hidden_dim, output_dim)
        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)
    def forward(self, x):
        x = torch.tanh(self.f1(x))
        mean = self.mean_layer(x)
        #log_std = self.log_std_layer(x)
        log_std = torch.zeros_like(mean)
        log_std.requires_grad = False
        std = torch.exp(log_std)
        std.requires_grad = False
        return mean, log_std, std

        
        
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_layer.weight.data.mul_(0.1)
        self.output_layer.bias.data.mul_(0.0)
    def forward(self, x):
        x = torch.tanh(self.f1(x))
        output = self.output_layer(x)
        return output