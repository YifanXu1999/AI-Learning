#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:54:18 2020

@author: yifanxu
"""
import torch
import torch.tensor as tensor

def flat_parameters(param):
    '''
    Convert a list of tensors with different sizes into an 1d array of parameters
    '''
    return torch.cat([grad.contiguous().view(-1) for grad in param])

def list_to_tensor(inp):
    '''
    Converts a list to a tensro list
    '''
    return torch.FloatTensor(inp)

def list_tensor_list_to_tensor(inp):
    '''
    Converts a list of tensor_list to tensor list
    '''
    return torch.cat(inp)
def list_tensor_to_tensor(inp):
    '''
    Convers a list of 0--d tensor to a 1d tensor list
    '''
    return torch.cat([x.view(-1) for x in inp])
    