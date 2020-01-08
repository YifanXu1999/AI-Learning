#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:19:50 2020

@author: yifanxu
"""

import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.autograd import grad
import math as m
from torch.nn.utils.convert_parameters import vector_to_parameters

env = gym.make('CartPole-v1')
seed = 543
env.seed(seed)
torch.manual_seed(seed)
gamma = 0.99
res_threshold = 1e-10
cg_max_iters = 30
delta = 0.00001

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(4, 20)
        self.l2 = nn.Linear(20, 2)
        self.drop = nn.Dropout(0.5)
        self.rewards = []
        # record of pi
        self.policy_prbs =  None
        # The list of probability of the selected actions along the trajectory
        self.selected_action_prb = []
        self.states = []
    
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.softmax(x, dim=1)
        return x
policy = Policy()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state.type(torch.DoubleTensor)
    probs = policy(state)
    if(policy.policy_prbs is None):
        policy.policy_prbs = probs
    else:
        policy.policy_prbs = torch.cat((policy.policy_prbs, probs), dim=0)
    m = Categorical(probs)
    action = m.sample()
    action_idx = action.item()
    policy.selected_action_prb.append(probs[0][action_idx])
    #print(probs[action.item()])
    return action.item()

def get_kl_compare(pi, pi_old):
    return (pi_old * torch.log(pi_old / pi)).mean().item()

def get_kl(pi):
    '''
    input: state
    This method computes the KL divergence given the input state, where
    kl(pi, pi_old) = mean of (pi_old * log (pi_old/pi), where pi = pi_old,
    grad of pi should be enabled, and grad of pi_old should be disabled
    '''
    pi_old = pi.data
    result = (pi_old * torch.log((pi_old / pi))).mean()
    return result

def get_Q():
    '''
    This method computes and returns the action-state values along the generated
    trajectory
    '''
    R = 0
    Q = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        Q.insert(0, R)
    return Q

def get_surrogate_loss():
    '''
    L = mean of (pi(a_n|s_n) / pi_old(a_n|sn) * Q_old(s_n, a_n)
    '''
    Q_old = get_Q()
    L_ = []
    for Q_sn_an_old, pi_sn_an in zip(Q_old, policy.selected_action_prb):
        pi_sn_an_old = pi_sn_an.data
        L_.append(pi_sn_an /  pi_sn_an_old *  Q_sn_an_old)
    L = L_[0]
    for i in range(1, len(L_)):
        L = L + L_[i]
    L = L / len(L_)
    return L

def flat_parameters(param):
    '''
    Convert a list of tensors with different sizes into an 1d array of parameters
    '''
    return torch.cat([grad.contiguous().view(-1) for grad in param])

def get_fisher_vector_product(x, damping = 1e-2):
    '''
    FVP is used to indirectly compute hassin matrix with more efficency, and it
    is used for conjugate gradient.
    y = Hx
    '''
    # Step 1, compute the product of first derivative of KL divergence wrt theta and x
    kl = get_kl(policy.policy_prbs)
    policy.zero_grad()
    kl_1_grads_ = torch.autograd.grad(kl, policy.parameters(), create_graph = True, retain_graph = True)
    kl_1_grads = flat_parameters(kl_1_grads_)
    # Step2, compute the sum of the product of kl first derivative and x
    kl_1_grads_product = kl_1_grads * x
    kl_1_grads_product_sum = kl_1_grads_product.sum()
    # Step3, obtain fisher_vector_product by differentiating the result we get at step2
    policy.zero_grad()
    kl_2_grads = torch.autograd.grad(kl_1_grads_product_sum, policy.parameters(), retain_graph = True)
    fisher_vector_product = flat_parameters(kl_2_grads)
    return fisher_vector_product + damping * x

def update_theta(theta, beta, s):
    '''
    This function computes and updates an appropriate theta, such that Dkl(pi, pi_old) < delta
    If with the current beta, the constraint doesnt hold, decresease the beta value exponentially
    '''
    beta_factor = 1
    beta_s = beta * s
    states = np.asarray(policy.states)
    states = torch.from_numpy(states).float().unsqueeze(0)
    before = 0
    with torch.no_grad():
        for i in range(100):
            new_theta = theta + beta_factor * beta_s
            #print(beta_factor * beta_s)
            vector_to_parameters(new_theta, policy.parameters())
            beta_factor = 1 / m.e
            pi = torch.cat([policy(state) for state in states])
            #print(get_kl_compare(pi, policy.policy_prbs))
            before = before - get_kl_compare(pi, policy.policy_prbs)
            if(get_kl_compare(pi, policy.policy_prbs) <= delta):
                break
            
            
        
        
    

def conjugate_gradient(b):
    '''
    input: b where Ax = b, A is the Fisher information matrix H, b is the gradient of the loss function
    Algorithm from wiki
    ---------------------------------------------------------------------------
    function x = conjgrad(A, b, x)
        r = b - A * x;
        p = r;
        rsold = r' * r;
    
        for i = 1:length(b)
            Ap = A * p;
            alpha = rsold / (p' * Ap);
            x = x + alpha * p;
            r = r - alpha * Ap;
            rsnew = r' * r;
            if sqrt(rsnew) < 1e-10
                  break;
            end
            p = r + (rsnew / rsold) * p;
            rsold = rsnew;
        end
    ---------------------------------------------------------------------------
    end
    '''
    # Init a x
    x = torch.zeros(b.size())
    # b - A * x = b because x = 0
    r = b.clone()
    p = r.clone()
    rsold = r.dot(r)
    for i in range(cg_max_iters):
        # A  = get_fisher_vector_product()
        Ap = get_fisher_vector_product(p)
        alpha = rsold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if m.sqrt(rsnew) < res_threshold:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew;
    return x

def update_policy():
    L  = get_surrogate_loss()
    L = L * -1
    policy.zero_grad()
    g_ = torch.autograd.grad(L, policy.parameters(), retain_graph=True)
    g = flat_parameters(g_)
    s = conjugate_gradient(g)
    s = s * -1
    Hs = get_fisher_vector_product(s, 0)
    sHs = s.dot(Hs)
   # print(s)
    # beta = Full step size, sqrt((2*eps) / (sHs) )
    beta =  m.sqrt(2*delta / sHs)
    theta = flat_parameters(policy.parameters())
    #print(beta, '\n', theta)
    #print(new_theta)
    #new_theta = line_search(theta, beta, s)
    #new_theta = theta + beta * s
    #vector_to_parameters(new_theta, policy.parameters())
    update_theta(theta, beta, s)
    policy.rewards = []
    policy.policy_prbs = None
    policy.selected_action_prb = []
    policy.states = []

def main():
    sum = 0
    for i in range(5000):
        state = env.reset()
        eps_reward = 0
        for t in range(500):
            #env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            eps_reward += reward
            policy.states.append(state)
            if(done):
                break
        print("Iter:", i, " ", eps_reward)
        update_policy()
        sum = sum * 0.95 +  eps_reward*0.05
    print(sum)
    #print(get_Q())
    #print(get_surrogate_loss())
if __name__ == '__main__':
    main()
    