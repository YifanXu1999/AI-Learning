#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:27:21 2020

@author: yifanxu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import math as m
from GAE import *
from Model import *
from torch.nn.utils.convert_parameters import vector_to_parameters

discount_factor = 0.99
trace_decay = 0.99
LEARNING_RATE = 0.01
cg_max_iters = 10
delta = 0.001
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def get_kl_compare(pi, pi_old):
    return (pi_old * torch.log(pi_old / pi)).mean().item()

def flat_parameters(param):
    '''
    Convert a list of tensors with different sizes into an 1d array of parameters
    '''
    return torch.cat([grad.contiguous().view(-1) for grad in param])

class Agent:
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        #action = Policy(input_dim, hidden_dim, output_dim)
        #value = Value(input_dim, hidden_dim)
        self.actor = Policy(input_dim, hidden_dim, output_dim)
        critic = Value(input_dim, hidden_dim)
        self.policy = ActorCritic(self.actor, critic)
        #self.policy.apply(init_weights)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        self.action_prbs = []
        self.states = []
    

    def select_action(self, state):
        action_prob, value_pred = self.policy(state)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        self.log_prob_actions.append(log_prob_action)
        self.values.append(value_pred)
        self.action_prbs.append(action_prob)
        
        return action
    
    def update_theta(self, theta, beta, s):
        '''
        This function computes and updates an appropriate theta, such that Dkl(pi, pi_old) < delta
        If with the current beta, the constraint doesnt hold, decresease the beta value exponentially
        '''
        beta_factor = 1
        beta_s = beta * s
        states = torch.cat(self.states)
        before = 0
        with torch.no_grad():
            for i in range(10):
                new_theta = theta + beta_factor * beta_s
                #print(beta_factor * beta_s)
                vector_to_parameters(new_theta, self.actor.parameters())
                beta_factor = 1 / m.e
                #pi = torch.cat([self.actor(state) for state in states])
                pi = self.actor.forward(states)
                #print(get_kl_compare(pi, policy.policy_prbs))
                before = before - get_kl_compare(pi, torch.cat(self.action_prbs))
                if(get_kl_compare(pi, torch.cat(self.action_prbs)) <= delta):
                    break
        
    #print(advantages)
    def get_kl(self, pi):
        '''
        input: state
        This method computes the KL divergence given the input state, where
        kl(pi, pi_old) = mean of (pi_old * log (pi_old/pi), where pi = pi_old,
        grad of pi should be enabled, and grad of pi_old should be disabled
        '''
        pi_old = pi.data
        result = (pi_old * torch.log((pi_old / pi))).mean()
        return result
    
    def get_fisher_vector_product(self, x, damping = 1e-2):
        '''
        FVP is used to indirectly compute hassin matrix with more efficency, and it
        is used for conjugate gradient.
        y = Hx
        '''
        # Step 1, compute the product of first derivative of KL divergence wrt theta and x
        kl = self.get_kl(torch.cat(self.action_prbs))
        self.actor.zero_grad()
        kl_1_grads_ = torch.autograd.grad(kl, self.actor.parameters(), create_graph = True, retain_graph = True)
        kl_1_grads = flat_parameters(kl_1_grads_)
        # Step2, compute the sum of the product of kl first derivative and x
        kl_1_grads_product = kl_1_grads * x
        kl_1_grads_product_sum = kl_1_grads_product.sum()
        # Step3, obtain fisher_vector_product by differentiating the result we get at step2
        self.actor.zero_grad()
        kl_2_grads = torch.autograd.grad(kl_1_grads_product_sum, self.actor.parameters(), retain_graph = True)
        fisher_vector_product = flat_parameters(kl_2_grads)
        return fisher_vector_product + damping * x
    
    def conjugate_gradient(self, A, b, res_threshold=1e-10):
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
            Ap = A(p)
            alpha = rsold / (p.dot(Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.dot(r)
            if m.sqrt(rsnew) < res_threshold:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew;
        return x
    
    def update_policy(self):
        rewards = self.rewards
        values = torch.cat(self.values).squeeze(-1)
        log_prob_actions = torch.cat(self.log_prob_actions)
        prob_actions = torch.pow(m.e, log_prob_actions)
        returns = calculate_returns(rewards, discount_factor)
        
        advantages = calculate_advantages(rewards, values, discount_factor, trace_decay)
        
        advantages = advantages.detach()
        returns = returns.detach()
        print(advantages)
        print(prob_actions)
        policy_loss = (advantages * (prob_actions / prob_actions.data)).sum()
        self.actor.zero_grad()
        g_ = torch.autograd.grad(policy_loss, self.actor.parameters(), retain_graph=True)
        g = flat_parameters(g_)
        s = self.conjugate_gradient(self.get_fisher_vector_product, g)
        print(s)
        Hs = self.get_fisher_vector_product(s, 0)
        sHs = s.dot(Hs)
        beta =  m.sqrt(2*delta / sHs)
        theta = flat_parameters(self.actor.parameters())
        self.update_theta(theta, beta, s)
        
        self.optimizer.zero_grad()
        value_loss = F.smooth_l1_loss(returns, values).sum()        
        value_loss.backward()
        
        self.optimizer.step()
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        self.action_prbs = []
        self.states = []
    