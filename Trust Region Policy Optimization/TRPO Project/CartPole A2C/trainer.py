#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:45:38 2020

@author: yifanxu
"""
import torch
from GAE import calculate_returns
from GAE import calculate_advantages
import torch.distributions as distributions
import torch.nn.functional as F
from utils import flat_parameters
import math as m
from torch.nn.utils.convert_parameters import vector_to_parameters
import torch.optim as optim

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

def get_fisher_vector_product(actor, x, action_prbs, damping = 1e-1):
    '''
    FVP is used to indirectly compute hassin matrix with more efficency, and it
    is used for conjugate gradient.
    y = Hx
    '''
    # Step 1, compute the product of first derivative of KL divergence wrt theta and x
    kl = get_kl(action_prbs)
    actor.zero_grad()
    kl_1_grads_ = torch.autograd.grad(kl, actor.parameters(), create_graph = True, retain_graph = True)
    kl_1_grads = flat_parameters(kl_1_grads_)
    # Step2, compute the sum of the product of kl first derivative and x
    kl_1_grads_product = kl_1_grads * x
    kl_1_grads_product_sum = kl_1_grads_product.sum()
    # Step3, obtain fisher_vector_product by differentiating the result we get at step2
    actor.zero_grad()
    kl_2_grads = torch.autograd.grad(kl_1_grads_product_sum, actor.parameters(), retain_graph = True)
    fisher_vector_product = flat_parameters(kl_2_grads)
    return fisher_vector_product + damping * x

def conjugate_gradient(actor, A, b, prob_actions, res_threshold=1e-10, cg_max_iters=10):
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
            Ap = A(actor, p, prob_actions)
            alpha = rsold / (p.dot(Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.dot(r)
            if m.sqrt(rsnew) < res_threshold:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew;
        return x

def train_actor(actor, states, actions, rewards, masks, values, kl_constraint=0.01):
    advantages = calculate_advantages(rewards, values, masks)
    action_probs = actor(states)
    dist = distributions.Categorical(action_probs)
    log_prob_actions = dist.log_prob(actions)
    advantages = advantages.detach()
    prob_actions = torch.pow(m.e, log_prob_actions)
    
    policy_loss = (advantages * (prob_actions / prob_actions.data)).sum()
    actor.zero_grad()
    g = torch.autograd.grad(policy_loss, actor.parameters(), retain_graph=True)
    g = flat_parameters(g)
    s = conjugate_gradient(actor, get_fisher_vector_product, g, prob_actions)
    Hs = get_fisher_vector_product(actor, s, prob_actions)
    sHs = s.dot(Hs)
    step_factor =  m.sqrt(2*kl_constraint / sHs)
    #print(step_factor)
    theta = flat_parameters(actor.parameters())
    new_theta = theta + step_factor * s
    actor.zero_grad()
    vector_to_parameters(new_theta, actor.parameters())
    
    
def train_critic(opt, states, actions, rewards, masks, values):
    returns = calculate_returns(rewards, masks)
    returns = returns.detach()
    value_loss = F.smooth_l1_loss(returns, values).sum() 
    opt.zero_grad()
    value_loss.backward()
    opt.step()