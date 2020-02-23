#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:01:15 2020

@author: yifanxu
"""

import torch

def flat_parameters(param):
    '''
    Convert a list of tensors with different sizes into an 1d array of parameters
    '''
    return torch.cat([grad.contiguous().view(-1) for grad in param])