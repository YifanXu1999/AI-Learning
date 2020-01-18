#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:34:28 2020

@author: yifanxu
"""


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import grad
from torch.distributions import Categorical
import math as m
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.distributions.multivariate_normal import MultivariateNormal
#------------------------------------------------------------------------------
seed = 543
env = gym.make('Swimmer-v2')
env.seed(seed)
torch.manual_seed(seed)

mean_range = 1
max_var = 0.1
gamma = 0.99
res_threshold = 1e-10
cg_max_iters = 10
delta = 0.01

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(8, 30)
        self.action_head_mean = nn.Linear(30, 2)
        self.action_head_var = nn.Linear(30, 2)
        self.rewards = []
        self.mu_records = None
        self.var_records = None
        self.x = None
        self.selected_action_prb = []
        self.states = []
    def forward(self, x):
        x = F.relu(self.l1(x))
        mu = F.tanh(self.action_head_mean(x)) * mean_range
        var = F.sigmoid(self.action_head_var(x)) * max_var
        return mu, var

def flat_parameters(param):
    '''
    Convert a list of tensors with different sizes into an 1d array of parameters
    '''
    return torch.cat([grad.contiguous().view(-1) for grad in param])

def get_kl_compare(mean1, var1, mean0, var0):
    std0 = torch.sqrt(var0)
    log_std0 = torch.log(std0)
    std1 = torch.sqrt(var1)
    log_std1 = torch.log(std1)
    kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def select_action(obs):
    state = torch.from_numpy(obs).float().unsqueeze(0).view(-1)
    mu, var = policy.forward(state)
    #print(mu, var)
    while(True):
        var_ = torch.tensor([[var[0], 0], [0, var[1]]])
        gaussin = MultivariateNormal(mu, var_)
        gauss_x = gaussin.sample()
        if (not any((x > 1 or x < -1)  for x in gauss_x)):
            break
    mu = mu.view(1, 2)
    var  = var.view(1,2)
    gauss_x_ = gauss_x.view(1,2)
    if policy.mu_records is None:
        policy.mu_records  = mu
        policy.var_records  = var
        policy.x = gauss_x_
    else:
        policy.mu_records  = torch.cat([policy.mu_records, mu], dim=0)
        policy.var_records = torch.cat([policy.var_records, var], dim=0)
        policy.x = torch.cat([policy.x, gauss_x_], dim=0)
    return gauss_x

def get_kl(mean1, var):
    std1 = torch.sqrt(var)
    log_std1 = torch.log(std1)
    mean0 = mean1.data
    log_std0 = log_std1.data
    std0 = std1.data
    kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def update_theta(theta, beta, s):
    '''
    This function computes and updates an appropriate theta, such that Dkl(pi, pi_old) < delta
    If with the current beta, the constraint doesnt hold, decresease the beta value exponentially
    '''
    beta_factor = 1
    beta_s = beta * s
    states = np.asarray(policy.states)
    states = torch.from_numpy(states).float().unsqueeze(0)
    #before = 0
    with torch.no_grad():
        for i in range(1):
            new_theta = theta + beta_factor * beta_s
            #print(beta_factor * beta_s)
            vector_to_parameters(new_theta, policy.parameters())
            beta_factor = 1 / m.e
            print(policy(state))
            #print(get_kl_compare(pi, policy.policy_prbs))
            #before = before - get_kl_compare(pi, policy.policy_prbs)
            #if(get_kl_compare(pi, policy.policy_prbs) <= delta):
               # break

def get_fisher_vector_product(x, damping = 1e-2):
    '''
    FVP is used to indirectly compute hassin matrix with more efficency, and it
    is used for conjugate gradient.
    y = Hx
    '''
    # Step 1, compute the product of first derivative of KL divergence wrt theta and x
    kl = get_kl(policy.mu_records, policy.var_records).mean()
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
        L_.append((pi_sn_an /  pi_sn_an_old)[:,0] *  Q_sn_an_old)
    L = sum(L_)
    L = L / len(L_)
    return L

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
    #print(sHs)
    beta =  m.sqrt(2*delta / sHs)
    theta = flat_parameters(policy.parameters())
    #print(beta, '\n', theta)
    #print(new_theta)
    #new_theta = line_search(theta, beta, s)
    #new_theta = theta + beta * s
    #vector_to_parameters(new_theta, policy.parameters())
    update_theta(theta, beta, s)
    policy.rewards = []
    policy.mu_records = None
    policy.var_records = None
    policy.x = None
    policy.selected_action_prb = []
    policy.states = []



policy = Policy()

def main():
    sum = 0
    for i in range(500):
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
        #print(policy.x)
        update_policy()
        sum = sum * 0.95 +  eps_reward*0.05
    print(sum)
    #print(get_Q())
    #print(get_surrogate_loss())
if __name__ == '__main__':
    main()