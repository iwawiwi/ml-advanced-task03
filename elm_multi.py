__author__ = 'iwawiwi'


from elm import elm
import numpy as np


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
num_hidden_neuron = 40
activation_function = 'radbas'
pseudo_inverse_method = 'svd'

#Y, TY = elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, pseudo_inverse_method)
# TODO: Swap training and testing data
Y, TY = elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, pseudo_inverse_method)

#print 'Y: ', Y
#print 'TY: ', TY

Y = np.reshape(Y, (len(attr1), -1))
TY = np.reshape(TY, (len(attr1), -1))

print 'Y: ', Y
print 'TY: ', TY

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, y_target, cmap = cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Target Function')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, Y, cmap = cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Train Result')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(attr1, attr2, TY, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('attr1')
ax.set_ylabel('attr2')
plt.title('Testing Result')
plt.show()

