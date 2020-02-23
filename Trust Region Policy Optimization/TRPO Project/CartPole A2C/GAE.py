#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:15:59 2020

@author: yifanxu
"""
import torch
import torch.nn.functional as F

def calculate_returns(rewards, masks, discount_factor=0.99):
    '''
    This function calculates the discounted rewards of the recorded
    trajectories generated from a number of paths
    input:
        rewards: 1d list
        masks: 1d list
        discount_factor = int
    output:
        returns: 1d tensor list
    '''
    returns = []
    R = 0
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + discount_factor * R * mask
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(rewards, values, masks, discount_factor=0.99, decay_factor=0.99):
    '''
    This function calculates the GAE values
    td_res(t) = r + discount_factor * v(t+1) - v(t)
    adv(t) = td_error + discount_factor * decay_factor * adv(t+1)
    input:
        rewards: 1d list
        values: 1d list
        masks: 1d list
        discount_factor: int
        decay_factor: int
    output:
        advnatges: 1d tensor list
    '''
    
    advantages = []
    adv = 0
    next_v = 0
    for r, v, mask in zip(reversed(rewards), reversed(values), reversed(masks)):
        td_res = r + discount_factor * mask * next_v - v
        adv = td_res + discount_factor * decay_factor * mask * adv
        next_v = v
        advantages.insert(0, adv)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages