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
    n = len(rewards)
    arr = np.arange(len(rewards))
    returns = calculate_returns(rewards, masks)
    values = critic.forward(torch.FloatTensor(states))
    #returns = returns.clone().reshape(values.size())
    criterion = torch.nn.MSELoss()
    advants = calculate_advantages(rewards, values, masks)
    for epoch in range(5):
        np.random.shuffle(arr)
        for i in range(n // 64):
            batch_index = arr[64 * i: 64 * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            target1 = returns.unsqueeze(1)[batch_index]
            target2 = advants.unsqueeze(1)[batch_index]
            values = critic.forward(torch.FloatTensor(states))
            values = values[batch_index]
            loss = criterion(values, target1 + target2)
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()