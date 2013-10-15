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
num_hidden_neuron = 9
pseudo_inverse_method = 'geninv'

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

sig_eta = t1 - t0
sine_eta = t2 - t1
hardlim_eta = t3 - t2
tribas_eta = t4 - t3
radbas_eta = t5 - t4



import matplotlib.pyplot as plt


###################################### SIGMOID AF ####################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + '][sigmoid][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SINC FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111)
# ax.plot(train_data[:,1], train_data[:,0], 'g-', linewidth=2, label='Train Data')
ax.scatter(train_data[:,1], train_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data')
ax.scatter(train_data[:,1], Y_sig, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Sigmoid AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training ELM with N data = ' + str(len(train_data)))
plt.legend()
plt.grid()

fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_sig, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Sigmoid AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(test_data)))
plt.legend()
plt.grid()


###################################### SINE AF #######################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + '][sine][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SINC FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111)
# ax.plot(train_data[:,1], train_data[:,0], 'g-', linewidth=2, label='Train Data')
ax.scatter(train_data[:,1], train_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data')
ax.scatter(train_data[:,1], Y_sine, s=10, facecolors='none', edgecolors='r', linewidths=0.5, label='Sine AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training ELM with N data = ' + str(len(train_data)))
plt.legend()
plt.grid()

fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_sine, s=10, facecolors='none', edgecolors='r', linewidths=0.5, label='Sine AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(test_data)))
plt.legend()
plt.grid()


###################################### HARDLIM AF ####################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + '][hardlim][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SINC FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111)
# ax.plot(train_data[:,1], train_data[:,0], 'g-', linewidth=2, label='Train Data')
ax.scatter(train_data[:,1], train_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data')
ax.scatter(train_data[:,1], Y_hardlim, s=10, facecolors='none', edgecolors='g', linewidths=0.5, label='HardLim AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training ELM with N data = ' + str(len(train_data)))
plt.legend()
plt.grid()


fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_hardlim, s=10, facecolors='none', edgecolors='g', linewidths=0.5, label='Hardlim AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(test_data)))
plt.legend()
plt.grid()


###################################### TRIBAS AF #####################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + '][tribas][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SINC FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111)
# ax.plot(train_data[:,1], train_data[:,0], 'g-', linewidth=2, label='Train Data')
ax.scatter(train_data[:,1], train_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data')
ax.scatter(train_data[:,1], Y_tribas, s=10, facecolors='none', edgecolors='m', linewidths=0.5, label='Tribas AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training ELM with N data = ' + str(len(train_data)))
plt.legend()
plt.grid()

fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_tribas, s=10, facecolors='none', edgecolors='m', linewidths=0.5, label='Tribas AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(test_data)))
plt.legend()
plt.grid()


###################################### RADBAS AF #####################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + '][radbas][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SINC FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111)
# ax.plot(train_data[:,1], train_data[:,0], 'g-', linewidth=2, label='Train Data')
ax.scatter(train_data[:,1], train_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data')
ax.scatter(train_data[:,1], Y_radbas, s=10, facecolors='none', edgecolors='c', linewidths=0.5, label='Radbas AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training ELM with N data = ' + str(len(train_data)))
plt.legend()
plt.grid()

fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY_radbas, s=10, facecolors='none', edgecolors='c', linewidths=0.5, label='Radbas AF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(test_data)))
plt.legend()
plt.grid()


################################ ACCURACY COMPARISON #################################
######################################################################################
title_str = '[h' + str(num_hidden_neuron) + '][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'TRAIN ACCURACY COMPARISON')
ax = fig.add_subplot(111)
cts = [Y_sig_acc, Y_sine_acc, Y_hardlim_acc, Y_tribas_acc, Y_radbas_acc]
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
plt.title('Error (RMSE) Comparison TRAIN ELM Hidden Neuron = ' + str(num_hidden_neuron))
plt.grid()

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
plt.title('Error (RMSE) Comparison TEST ELM Hidden Neuron = ' + str(num_hidden_neuron))
plt.grid()

plt.show()