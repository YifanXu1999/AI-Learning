#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:32:12 2020

@author: yifanxu
"""
import torch
import torch.nn as n
import torch.nn.functional as F
from utils.GAE import calculate_advantages
from utils.GAE import calculate_returns
from torch.distributions import Normal
import torch.autograd as autograd
from TRPO import fisher_vector_product
from TRPO import conjugate_gradient
from TRPO import get_kl
from utils.helpers import flat_parameters
import math as m
from torch.nn.utils.convert_parameters import vector_to_parameters

def get_surrogate_loss(advantages, actions, mu, std):
    #print(actions.size())
    #print(mu.size())
    #print(std.size())
    normal = Normal(mu, std)
    pi_new = torch.exp(normal.log_prob(actions))
    pi_old = pi_new.clone().detach()
    pi_new_over_pi_old = (pi_new / pi_old).sum(dim=1)
    L = (pi_new_over_pi_old * advantages).mean()
    return L

def update_theta(actor, old_theta, full_step, mu, std, states, constraint, max_iter = 5):
    factor = 1
    for i in range(max_iter):
        new_theta = old_theta + factor * full_step
        vector_to_parameters(new_theta, actor.parameters())
        mu_new, _, std_new = actor.forward(states)
        kl = get_kl(mu, std, mu_new, std_new)
        print(i, 'th search of kl', float(kl))
        if(float(kl) > constraint):
            factor = factor / m.e
        else:
            break

def train_actor(actor, critic, states, actions, rewards, masks, constraint_val=0.01):
    '''
    This function trains the input actor
    input:
        actor
        critic
        states: 1d list
        actions: tensor matrix
        reward: 1d list
        masks: 1d list
    '''
    states = torch.FloatTensor(states)
    # Get mu, std, and advantages
    mu, _, std = actor.forward(states)
    #advantages = calculate_advantages(rewards, values, masks)
    # Get the surrogate loss
    returns = calculate_returns(rewards, masks)
    values = critic(states)
    advantages = calculate_advantages(rewards, values, masks)
    #print(advantages)
    loss = get_surrogate_loss(returns, actions, mu, std)
    # Get the gradient of loss over  (g)
    actor.zero_grad()
    g = autograd.grad(loss, actor.parameters(), retain_graph=True)
    g = flat_parameters(g)
    # Get the direction of improvement using the conjugate gradient (s)
    s = conjugate_gradient(actor, mu, std, g.data)
    # Get Hs
    Hs = fisher_vector_product(actor, mu, std, s)
    # Get sHs
    sHs = s.dot(Hs)
    # Calculate the step factor
    step_factor = m.sqrt(2 * constraint_val / sHs)
    # Get the full step of theta improvement
    full_step = s * step_factor
    # Update theta
    old_theta = flat_parameters(actor.parameters())
    update_theta(actor, old_theta, full_step, mu, std, states, constraint_val)