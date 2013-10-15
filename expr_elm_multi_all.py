__author__ = 'iwawiwi'


from elm import elm
import numpy as np
from time import time


def true_fun(var1, var2):
    return np.sin(var1) + np.cos(var2)

# true signal
x1 = x2 = np.linspace(0.0, 2 * np.pi, 100, True)
x1, x2 = np.meshgrid(x1, x2)
y_true = true_fun(x1, x2)


# Initialize sample size
# It is the actual target
n_sample = 100
attr1 = attr2 = np.linspace(0.0, 2*np.pi, n_sample, True)
attr1, attr2 = np.meshgrid(attr1, attr2)
y_target = true_fun(attr1, attr2)

#print 'attr1: ', attr1.shape
#print 'attr2: ', attr2.shape
#print 'y_target: ', y_target.shape

attr1_flat = np.squeeze(np.asarray(np.mat(attr1).flatten().T))
attr2_flat = np.squeeze(np.asarray(np.mat(attr2).flatten().T))
y_target_flat = np.squeeze(np.asarray(np.mat(y_target).flatten().T))

#print 'attr1_flat: ', attr1_flat.shape
#print 'attr2_flat: ', attr2_flat.shape
#print 'y_target_flat: ', y_target_flat.shape

# noisy signal for test data
noise1 = [np.random.normal(0, 0.01, len(x)) for x in attr1]
signal = true_fun(attr1, attr2)
y_test1 = signal + noise1

y_test1_flat = np.squeeze(np.asarray(np.mat(y_test1).flatten().T))

"""
This is training data construction
"""
train_data = np.zeros(shape=(n_sample*n_sample,3))
train_data[:,0] = y_target_flat
train_data[:,1] = attr1_flat
train_data[:,2] = attr2_flat
train_data = np.mat(train_data) # cast as matrix
"""
This is testing data construction
"""
# test data == y_test1 + y_test2
test_data = np.zeros(shape=(n_sample*n_sample,3))
test_data[:,0] = y_test1_flat
test_data[:,1] = attr1_flat
test_data[:,2] = attr2_flat
test_data = np.mat(test_data) # cast as matrix

REGRESSION = 0
elm_type = REGRESSION
num_hidden_neuron = 50
#activation_function = 'radbas'
pseudo_inverse_method = 'svd'

t0 = time()
Y_sig, TY_sig, Y_sig_acc, TY_sig_acc = elm(train_data, test_data, elm_type, num_hidden_neuron, 'sigmoid', pseudo_inverse_method)
t1 = time()
Y_sine, TY_sine, Y_sine_acc, TY_sine_acc  = elm(train_data, test_data, elm_type, num_hidden_neuron, 'sine', pseudo_inverse_method)
t2 = time()
Y_hardlim, TY_hardlim, Y_hardlim_acc, TY_hardlim_acc  = elm(train_data, test_data, elm_type, num_hidden_neuron, 'hardlim', pseudo_inverse_method)
t3 = time()
Y_tribas, TY_tribas, Y_tribas_acc, TY_tribas_acc  = elm(train_data, test_data, elm_type, num_hidden_neuron, 'tribas', pseudo_inverse_method)
t4 = time()
Y_radbas, TY_radbas, Y_radbas_acc, TY_radbas_acc  = elm(train_data, test_data, elm_type, num_hidden_neuron, 'radbas', pseudo_inverse_method)
t5 = time()


# Time elapsed
sig_eta = t1 - t0
sine_eta = t2 - t1
hardlim_eta = t3 - t2
tribas_eta = t4 - t3
radbas_eta = t5 - t4

# RESHAPE matrix
Y_sine = np.reshape(Y_sine, (len(attr1), -1))
TY_sine = np.reshape(TY_sine, (len(attr1), -1))
Y_sig = np.reshape(Y_sig, (len(attr1), -1))
TY_sig = np.reshape(TY_sig, (len(attr1), -1))
Y_hardlim = np.reshape(Y_hardlim, (len(attr1), -1))
TY_hardlim = np.reshape(TY_hardlim, (len(attr1), -1))
Y_tribas = np.reshape(Y_tribas, (len(attr1), -1))
TY_tribas = np.reshape(TY_tribas, (len(attr1), -1))
Y_radbas = np.reshape(Y_radbas, (len(attr1), -1))
TY_radbas = np.reshape(TY_radbas, (len(attr1), -1))

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(attr1, attr2, y_target, cmap = cm.hot, alpha=0.3)
#ax.set_zlim(-2.5, 2.5)
#ax.set_xlabel('attr1')
#ax.set_ylabel('attr2')
#plt.title('Target Function')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(attr1, attr2, Y_sig, cmap = cm.hot, alpha=0.3)
#ax.set_zlim(-2.5, 2.5)
#ax.set_xlabel('attr1')
#ax.set_ylabel('attr2')
#plt.title('Train Result')
# plt.show()

title_str = '[h' + str(num_hidden_neuron) + '][sigmoid][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SIN+COS FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, TY_sig, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Testing Result')

title_str = '[h' + str(num_hidden_neuron) + '][sine][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SIN+COS FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, TY_sine, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Testing Result')

title_str = '[h' + str(num_hidden_neuron) + '][hardlim][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SIN+COS FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, TY_hardlim, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Testing Result')

title_str = '[h' + str(num_hidden_neuron) + '][tribas][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SIN+COS FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, TY_tribas, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Testing Result')

title_str = '[h' + str(num_hidden_neuron) + '][radbas][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SIN+COS TRAIN REGRESSION')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, TY_radbas, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Testing Result')


############################ TESTING ACC ##########################################
title_str = '[h' + str(num_hidden_neuron) + '][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'TEST ACCURACY COMPARISON')
ax = fig.add_subplot(111)
cts = [TY_sig_acc, TY_sine_acc, TY_hardlim_acc, TY_tribas_acc, TY_radbas_acc]
b = [0.15, 0.35, 0.55, 0.75, 0.95]
plt.xlim(0.0, 1.3)
tick_offset = [0.05] * 5
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('Sigmoid', 'Sine', 'Hardlim', 'Tribas', 'Radbas'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Activatiom Function')
plt.ylabel('RMSE')
plt.title('Error (RMSE) Comparison TEST ELM with Hidden Neuron = ' + str(num_hidden_neuron))
plt.grid()


############################ TIME COMPARISON ######################################
title_str = '[h' + str(num_hidden_neuron) + '][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'TEST TIME COMPARISON')
ax = fig.add_subplot(111)
cts = [sig_eta, sine_eta, hardlim_eta, tribas_eta, radbas_eta]
b = [0.15, 0.35, 0.55, 0.75, 0.95]
plt.xlim(0.0, 1.3)
tick_offset = [0.05] * 5
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('Sigmoid', 'Sine', 'Hardlim', 'Tribas', 'Radbas'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Activatiom Function')
plt.ylabel('Time (s)')
plt.title('Time Elapsed to TEST ELM with Hidden Neuron = ' + str(num_hidden_neuron))
plt.grid()



plt.show()

