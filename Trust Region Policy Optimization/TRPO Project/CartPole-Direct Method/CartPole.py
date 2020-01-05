#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 22:49:48 2019

@author: yifanxu
"""

import argparse
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

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)
gamma = 0.99
delta = 0.0000001

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(4, 20)
        self.l2 = nn.Linear(20, 2)
        self.rewards = []
        self.prbs =  None
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
states = torch.ones((1, 4))
eps = np.finfo(np.float32).eps.item()

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
def tensor_flat(tensors):
    ret = None
    for tensor in tensors:
        if ret is None:
            ret = tensor.view(-1)
        else:
            ret = torch.cat((ret, tensor.view(-1)), dim=0)
    return ret
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    if(policy.prbs is None):
        policy.prbs = probs
    else:
        policy.prbs = torch.cat((policy.prbs, probs), dim=0)
    m = Categorical(probs)
    action = m.sample()
    action_idx = action.item()
    policy.selected_action_prb.append(probs[0][action_idx])
    #print(probs[action.item()])
    return action.item()

def get_hessian():
    kl = get_kl(policy.prbs)
    # Get the jacobian matrix,
    policy.zero_grad()
    J_tmp = torch.autograd.grad(kl, policy.parameters(), create_graph=True, retain_graph=True)
    J = None
    # Concate
    for grad in J_tmp:
        if J is None:
            J = grad.view(-1)
        else:
            J = torch.cat((J, grad.view(-1)), dim=0)
    H_ = None
    for J_i in J:
        policy.zero_grad()
        H_i_ = torch.autograd.grad(J_i, policy.parameters(), create_graph=False, retain_graph=True)
        H_i = torch.cat([grad.contiguous().view(-1) for grad in H_i_])
        if H_ is None:
            H_ = H_i
        else:
            H_ = torch.cat((H_, H_i), 0)
    
    n_X_n = list(H_.size())[0]
    n = int(m.sqrt(n_X_n))
    H = H_.view(n, n)
    return H
    #print(J)

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

def update_policy():
    # Solve Hs = g to get s
    # Step 1:
    L  = get_surrogate_loss()
    policy.zero_grad()
    g_ = torch.autograd.grad(L, policy.parameters(), retain_graph=True)
    g = tensor_flat(g_)
    H = get_hessian()
    s = H.pinverse() @ g
    # Calculate beta, b = sqrt(2 *delta / sHs)
    s_t = s.view(s.size()[0], 1)
    sHs = s @ H @ s_t
    beta = ((2 * delta) / sHs).sqrt().item()
    states = np.asarray(policy.states)
    states = torch.from_numpy(states).float().unsqueeze(0)
    with torch.no_grad():
        l = []
        for param in policy.parameters():
            #print(param.view(-1))
            l.append(param.view(-1))
        theta_old = torch.cat(l)
        #print(theta_old)
        for count in range(10):
            i = 0
            theta = theta_old + beta * s
            for param in policy.parameters():
                size = np.array(param.size()).prod()
                param.data = theta[i:size + i].reshape(param.size())
                i += size
            pi = torch.cat([policy(state) for state in states])
            if(get_kl_compare(pi, policy.prbs) <= 0.001):
                break
            else:
                beta = beta / 3
        #print(theta)
    # 
    # Re-initalize
    policy.rewards = []
    policy.prbs =  None
    # The list of probability of the selected actions along the trajectory
    policy.selected_action_prb = []
    policy.states = []
    
    
def main():
    sum = 0
    for i in range(1000):
        state = env.reset()
        reward = 0
        ep_reward = 0
        for t in range(1000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            policy.states.append(state)
            #print(policy.states )
            ep_reward += reward
            if done:
                break
        states = np.asarray(policy.states)
        states = torch.from_numpy(states).float().unsqueeze(0)
        #print(policy.policy_his)
        #print(get_hessian(policy.prbs))
        update_policy()
        print(ep_reward)
        sum = sum * 0.95 + 0.05 *ep_reward
    print(sum)
#print(hessian(hessian(states)))
#kl = get_kl(states)
#J = grad(kl, policy.parameters(), create_graph=True, retain_graph=True)
#kl = get_kl(states)
#J = torch.autograd.grad(kl, policy.parameters(), create_graph=True, retain_graph=True)
#x2 = torch.Tensor([1])
#x1.requires_grad = True
##y = x1 ** 3 + x2 * x1
#y.backward()
#J =  torch.autograd.grad(y, [x1, x2], create_graph=True, retain_graph=True)
#print(J)
#print(hessian1(J, [x1, x2]))
#print(get_kl(states))
#print(J)
if __name__ == '__main__':
    main()
    
#hessian(torch)
#print(env.step(env.action_space.sample()))
