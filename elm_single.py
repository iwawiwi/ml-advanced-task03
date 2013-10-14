__author__ = 'iwawiwi'


from elm import elm
import numpy as np

# TODO: This is self made data
# true signal
x = np.linspace(0.0, 2*np.pi, 100, True)
y_true = np.sin(x)

# Initialize sample size
# It is the actual target
n_sample = 1000
attr1 = np.linspace(0.0, 2*np.pi, n_sample, True)
y_target = np.sin(attr1)

# noisy signal for test data
noise1 = np.random.normal(0, 0.10, len(attr1))
noisy_attr = attr1 + noise1
y_test1 = np.sin(noisy_attr)

"""
This is training data construction
"""
# train data == y_sample1 + y_sample2
train_data = np.zeros(shape=(n_sample,2))
train_data[:,0] = y_target
train_data[:,1] = attr1
train_data = np.mat(train_data) # cast as matrix
"""
This is testing data construction
"""
# test data == y_test1 + y_test2
test_data = np.zeros(shape=(n_sample,2))
test_data[:,0] = y_test1
test_data[:,1] = noisy_attr
test_data = np.mat(test_data) # cast as matrix

REGRESSION = 0
elm_type = REGRESSION
num_hidden_neuron = 5
activation_function = 'tribas'
pseudo_inverse_method = 'svd'

Y, TY = elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, pseudo_inverse_method)

import matplotlib.pyplot as plt

fig = plt.figure('TRAIN REGRESSION')
ax = fig.add_subplot(111)
ax.plot(x, y_true, 'g-', linewidth=2, label='True')
ax.scatter(attr1, y_target, s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data')
#ax.scatter(attr1, Y, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Train Result')
ax.plot(attr1, Y, 'y-', linewidth=2, label='Train Output')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(attr1)))
plt.legend()
plt.grid()
# plt.show()

fig = plt.figure('TEST REGRESSION')
ax = fig.add_subplot(111)
ax.plot(x, y_true, 'g-', linewidth=2, label='True')
ax.scatter(noisy_attr, y_test1, s=10, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data')
#ax.scatter(noisy_attr, TY, s=10, facecolors='none', edgecolors='y', linewidths=0.5, label='Test Result')
ax.plot(noisy_attr, TY, 'y-', linewidth=2, label='Test Output')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(attr1)))
plt.legend()
plt.grid()
plt.show()