#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:42:18 2020å

@author: yifanxu
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
from utils import *
from model import Policy
import math as m
import gym
from torch.distributions import Normal
from TrpoHelpers import *
from torch.nn.utils.convert_parameters import vector_to_parameters

seed = 543
env = gym.make('Swimmer-v2')
env.seed(seed)
torch.manual_seed(seed)

class Agent:
    
    def __init__(self, policy):
        self.policy = policy
    
    def select_action(self, obs):
        '''
        Input:
            State: list
        Sample the action, and store the (mu, std), and the density of the action
        '''
        state = torch.from_numpy(obs).float().unsqueeze(0)
        self.policy.states.append(state)
        mu, _, std = self.policy.forward(state)
        self.policy.mu_records.append(mu)
        self.policy.std_records.append(std)
        # Build the gaussin distribution
        distribution = Normal(mu, std)
        action = distribution.sample()
        density = torch.pow(m.e, distribution.log_prob(action))
        self.policy.action_prb_records.append(density)
        return action

    def search_theta(self, full_step, beta):
        theta_ = self.policy.parameters()
        theta = flat_parameters(theta_)
        states = list_tensor_list_to_tensor(self.policy.states)
        mu_old, _, std_old = self.policy.forward(states)
        d_old = Normal(mu_old, std_old)
        step = beta * full_step
        step_factor = 1
        for i in range(10):
            with torch.no_grad():
                new_theta = theta + step_factor * step
                vector_to_parameters(new_theta, self.policy.parameters())
                mu_new, _, std_new = self.policy.forward(states)
                d_new = Normal(mu_new, std_new)
                step_factor = step_factor / 3
                if(get_kl_diff(d_old, d_new) <= self.policy.delta):
                    break
            

    def update_policy(self):
        '''
        Update the policy
        '''
        Q_ = get_Q(self.policy.gamma, self.policy.rewards)
        Q = list_tensor_to_tensor(Q_)
        L = get_surrogate_loss(list_tensor_list_to_tensor(self.policy.action_prb_records), Q)
        self.policy.zero_grad()
        g_ = torch.autograd.grad(L, self.policy.parameters(), retain_graph=True)
        g = flat_parameters(g_)
        mu = list_tensor_list_to_tensor(self.policy.mu_records)
        std = list_tensor_list_to_tensor(self.policy.std_records)
        kl = get_kl(mu, std)
        s_ = conjugate_gradient(self.policy, kl, g)
        Hs = get_fisher_vector_product(self.policy, kl, s_, 0)
        sHs = s_.dot(Hs)
        beta =  m.sqrt(2 * self.policy.delta / sHs)
        self.search_theta(s_, beta)
        
    def reset_policy_records(self):
        self.action_prb_records = []
        self.mu_records = []
        self.std_records = []
        self.rewards = []
        self.states = []
        self.policy.zero_grad()
'''
policy = Policy(8, 2)
agent = Agent(policy)
state = env.reset()
agent.select_action(state)
agent.select_action(state)
print(policy.action_prb_records)
print(list_tensor_to_tensor(policy.action_prb_records))
mu = list_tensor_to_tensor(policy.mu_records)
std = list_tensor_to_tensor(policy.std_records)
print(get_kl(mu, std).mean())
'''