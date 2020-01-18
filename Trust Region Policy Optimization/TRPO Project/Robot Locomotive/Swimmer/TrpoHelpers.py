#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:42:10 2020

@author: yifanxu
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
from utils import *
from model import Policy
import math as m
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import numpy as np

def get_surrogate_loss(pi, Q):
    '''
    Input:
        pi: tensor list
        Q: tensor list, 1d
    '''
    pi_old = pi.data.clone()
    ratio = (pi / pi_old).mean(dim=1)
    return (ratio @ Q)

def get_kl_diff(d1, d2):
    '''
    Get the kl diff of d1, d2
    '''
    return kl_divergence(d1, d2).mean()

def get_kl(mu, std):
    '''
    Input:
        mu: tensor list
        std: tensor list
    This function computes the Kl divergence of the inputs states.
    Where kl = pi.data * log(pi.data / pi)
    '''
    q = Normal(mu, std)
    p = Normal(mu.clone().data.clone(), std.data.clone())
    return kl_divergence(p, q).mean()

def get_Q(gamma, rewards):
    '''
    Input:
        gamma: float
        rewards: list
    This method computes and returns the action-state values along the generated
    trajectory
    '''
    R = 0
    Q = []
    for r in rewards[::-1]:
        R = r + gamma * R
        Q.insert(0, R)
    return Q

def conjugate_gradient(policy, kl_div, b, cg_max_iters=10, res_threshold=1e-10):
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
        Ap = get_fisher_vector_product(policy, kl_div, p)
        alpha = rsold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if m.sqrt(rsnew) < res_threshold:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew;
    return x

def get_fisher_vector_product(policy, kl_div, x, damping = 1e-2):
    '''
    FVP is used to indirectly compute hassin matrix with more efficency, and it
    is used for conjugate gradient.
    y = Hx
    '''
    # Step 1, compute the product of first derivative of KL divergence wrt theta and x
    kl = kl_div.clone()
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

#def get_surrogate_loss(mu, std, x, Q):
