__author__ = 'iwawiwi'


from elm import elm
import numpy as np

train_data = np.loadtxt('sinc_train', delimiter=' ')
test_data = np.loadtxt('sinc_test', delimiter=' ')

#print train_data.shape, train_data
#print test_data.shape, test_data

REGRESSION = 0
elm_type = REGRESSION
num_hidden_neuron = 9
activation_function = 'sig'
pseudo_inverse_method = 'svd'

Y, TY, Y_acc, T_acc = elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, pseudo_inverse_method)

import matplotlib.pyplot as plt

title_str = '[test][h' + str(num_hidden_neuron) + '][' + activation_function + '][' + pseudo_inverse_method + ']'
fig = plt.figure(title_str + 'SINC FUNCTION TRAIN REGRESSION')
ax = fig.add_subplot(111)
# ax.plot(train_data[:,1], train_data[:,0], 'g-', linewidth=2, label='Train Data')
ax.scatter(train_data[:,1], train_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data')
ax.scatter(train_data[:,1], Y, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Train Output')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training ELM with N data = ' + str(len(train_data)))
plt.legend()
plt.grid()
# plt.show()

fig = plt.figure(title_str + 'SINC FUNCTION TEST REGRESSION')
ax = fig.add_subplot(111)
ax.scatter(test_data[:,1], test_data[:,0], s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
ax.scatter(test_data[:,1], TY, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Test Output')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(test_data)))
plt.legend()
plt.grid()
plt.show()