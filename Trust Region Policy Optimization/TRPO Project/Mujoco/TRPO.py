#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:41:46 2020

@author: yifanxu
"""
from torch.distributions import Normal
import torch.distributions.kl as kl
import torch
from  utils.helpers import flat_parameters
import torch.nn as nn
import math as m
import torch.autograd as autograd


def kl_divergence(mu, std, mu_old, std_old):
    normal_new = Normal(mu, std)
    normal_old = Normal(mu_old, std_old)
    D_kl = kl.kl_divergence(normal_old, normal_new)
    D_kl = D_kl.sum(dim=1)
    return D_kl.mean()

def get_kl(mu_old, std_old, mu_new, std_new):
    '''
    Computes the kl divergences between the two distribution models shaped by (mu_old, std_old) and
    (mu_new, std_new) respectively. Dkl (pi_old, pi_new) = pi_old * log (pi_old / pi_new)
    kl(p, q) = p * log (p / q)
    input:
        mu_old: tensor matrix
        std_old: tensor matrix
        mu_new: tensor matrix
        std_new: tensor matrix
    '''
    #p = Normal(mu_old, std_old)
    #q = Normal(mu_new, std_new)
    #D_kl = kl.kl_divergence(p, q)
    #return D_kl.sum(dim=1).mean()
    D_kl = kl_divergence(mu_new, std_new, mu_old, std_old)
    return D_kl.mean()

def fisher_vector_product(actor, mu, std, x, dampling=1e-1):
    '''
    input:
        mu: tensor matrix
        std: tensor matrix
    '''
    D_kl = get_kl(mu.detach().clone(), std.detach().clone(), mu, std)
    # Get first order grad
    actor.zero_grad()
    first_order_grads = autograd.grad(D_kl, actor.parameters(), create_graph=True, retain_graph=True)
    first_order_grads = flat_parameters(first_order_grads)
    # Multiply first_order_grads with x, and sum it up
    first_order_grads_x = (first_order_grads * x).sum()
    # Get the second order grad
    actor.zero_grad()
    second_order_grads = autograd.grad(first_order_grads_x, actor.parameters(), retain_graph=True)
    second_order_grads = flat_parameters(second_order_grads)
    return second_order_grads + x * dampling

def conjugate_gradient(actor, mu, std, b, cg_max_iters=10, res_threshold=1e-10):
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
    rsold = torch.dot(r, r)
    for i in range(cg_max_iters):
        # A  = get_fisher_vector_product()
        Ap = fisher_vector_product(actor, mu, std, p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if m.sqrt(rsnew) < res_threshold:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew;
    return x
