#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:31:55 2020

@author: yifanxu
"""
import torch
import torch.nn.functional as F
from utils.GAE import calculate_returns
from utils.GAE import calculate_advantages
import numpy as np
def train_critic(critic, critic_optimizer, states, rewards, masks):
    '''
    This function trains the critic
    input:
        critic_optimizer
        rewards: 1d list
        masks: 1d list
        values: 1d tesnror
    output:
        None
    '''
    states = torch.FloatTensor(states)
    values = critic(states).reshape(-1)
    advantages = calculate_advantages(rewards, values, masks)
    returns = calculate_returns(rewards, masks)
    advantages = advantages.detach()
    returns = returns.detach()
    value_loss = F.smooth_l1_loss(returns, values).sum()
    critic_optimizer.zero_grad()
    value_loss.backward()
    critic_optimizer.step()