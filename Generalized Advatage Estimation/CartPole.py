#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:13:42 2020

@author: yifanxu
Refer to https://github.com/bentrevett/pytorch-rl/blob/master/4%20-%20Generalized%20Advantage%20Estimation%20(GAE).ipynb
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import gym