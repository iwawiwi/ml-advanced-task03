__author__ = 'iwawiwi'

import numpy as np
"""
http://www.dobots.nl/blog/-/blogs/extreme-learning-machines
"""

num_sample = 100 # number of training sample
gen = np.round(0.2 * num_sample) # generalisation sample
in_dim = 2 # Input dimension
out_dim = 1 # Output dimension

x = np.mat(np.round(np.random.rand(in_dim,num_sample+gen))) # input data
y = np.mat(np.zeros(shape=(out_dim,num_sample+gen))) # output data

y = np.logical_xor(x[1,:], y[2,:]) # perform logical XOR
y.astype(int) # cast to INT

h = 4 # amount of hidden nodes
SH = np.random.rand(h,in_dim) # input-to-hidden synaptic weights, fixed
BH = np.random.rand(h,1) * np.ones(1,num_sample+gen) # hidden layer bias, fixed
S = np.zeros(out_dim,h) # hidden-to-output synaptic weights, to be adapted

H = np.tanh(-BH + SH * x) # calculate hidden layer output matrix
# TODO: INCOMPLETE!