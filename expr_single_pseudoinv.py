__author__ = 'iwawiwi'


from elm import elm
import numpy as np
from time import time

train_data = np.loadtxt('sinc_train', delimiter=' ')
test_data = np.loadtxt('sinc_test', delimiter=' ')

#print train_data.shape, train_data
#print test_data.shape, test_data

REGRESSION = 0
elm_type = REGRESSION
num_hidden_neuron = 10
activation_function = 'sigmoid'

t0 = time()
Y_svd, TY_svd, Y_svd_acc, TY_svd_acc = elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, 'svd')
t1 = time()
Y_geninv, TY_geninv, Y_geninv_acc, TY_geninv_acc  = elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, 'geninv')
t2 = time()
Y_qr, TY_qr, Y_qr_acc, TY_qr_acc  = elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, 'qrpivot')
t3 = time()

svd_eta = t1 - t0
geninv_eta = t2 - t1
qr_eta = t3 - t2


import matplotlib.pyplot as plt


###################################### SIGMOID AF ####################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + '][sigmoid][svd]'
fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_svd, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Sigmoid AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('ELM Test with Pseudo-Inverse : Moore-Penrose')
plt.legend()
plt.grid()

title_str = '[h' + str(num_hidden_neuron) + '][sigmoid][geninv]'
fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_geninv, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Sigmoid AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('ELM Test with Pseudo-Inverse : Fast Moore-Penrose (GENINV)')
plt.legend()
plt.grid()

title_str = '[h' + str(num_hidden_neuron) + '][sigmoid][qr]'
fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_qr, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Sigmoid AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('ELM Test with Pseudo-Inverse : QR Decomposition (QR)')
plt.legend()
plt.grid()

################################ ACCURACY COMPARISON #################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + ']'
fig = plt.figure(title_str + 'TRAIN ACCURACY COMPARISON')
ax = fig.add_subplot(111)
cts = [TY_svd_acc, TY_geninv_acc, TY_qr_acc]
b = [0.15, 0.35, 0.55]
plt.xlim(0.0, 0.8)
tick_offset = [0.05] * 3
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('Moore-Penrose', 'GENINV', 'QR'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Pseudo-Inverse Method')
plt.ylabel('RMSE')
plt.title('Error (RMSE) Comparison TRAIN ELM Hidden Neuron = ' + str(num_hidden_neuron))
plt.grid()

############################ TIME COMPARISON ##########################################
title_str = '[h' + str(num_hidden_neuron) + ']'
fig = plt.figure(title_str + 'TIME COMPARISON')
ax = fig.add_subplot(111)
cts = [svd_eta, geninv_eta, qr_eta]
b = [0.15, 0.35, 0.55]
plt.xlim(0.0, 0.8)
tick_offset = [0.05] * 3
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('Moore-Penrose', 'GENINV', 'QR'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Pseudo-Inverse Method')
plt.ylabel('Time (s)')
plt.title('Time Comparison TEST ELM Hidden Neuron = ' + str(num_hidden_neuron))
plt.grid()

plt.show()