__author__ = 'iwawiwi'


import numpy as np
from time import time
import utils
import matplotlib.pyplot as plt


# true signal
x = np.linspace(0.0, 2*np.pi, 100, True)
y_true = np.sin(x)


# noisy signal
"""
Initialize sample size
"""
n_sample = 10000
x_sample = np.linspace(0.0, 2*np.pi, n_sample, True)
noise1 = np.random.normal(0, 0.15, len(x_sample))
y_sample1 = np.sin(x_sample) + noise1
noise2 = np.random.normal(0,0.2, len(x_sample))
y_sample2 = np.sin(x_sample) + noise2

# noisy signal for test data
noise3 = np.random.normal(0, 0.10, len(x_sample))
y_test1 = np.sin(x_sample) + noise3
noise4 = np.random.normal(0,0.25, len(x_sample))
y_test2 = np.sin(x_sample) + noise4


"""
This is training data construction
"""
# train data == y_sample1 + y_sample2
train_data = np.zeros(shape=(n_sample,2))
train_data[:,0] = y_sample1
train_data[:,1] = y_sample2
train_data = np.mat(train_data) # cast as matrix
"""
This is testing data construction
"""
# test data == y_test1 + y_test2
test_data = np.zeros(shape=(n_sample,2))
test_data[:,0] = y_test1
test_data[:,1] = y_test2
test_data = np.mat(test_data) # cast as matrix
"""
INPUT Parameter:
Elm_Type = 0 for regression and 1 for classification
NumberofHiddenNeurons
ActivationFunction
PseudoInverseMethod
"""
REGRESSION = 0
CLASSIFIER = 1
elm_type = REGRESSION
num_hidden_neuron = 1
activation_function = 'sine'
pseudo_inverse_method = 'svd'


##################################################################
######################## LOAD TRAINING DATA SET ##################
T = train_data[:,0].T
P = train_data[:,1:np.size(train_data,1)].T


##################################################################
######################## LOAD TESTING DATA SET ###################
TVT = test_data[:,0].T
TVP = test_data[:,1:np.size(test_data,1)].T

"""
Initialize NUMBER of NEURON, TEST DATA, and TRAIN DATA
"""
num_train_data = np.size(P,1)
num_test_data = np.size(TVP,1)
num_input_neuron = np.size(P,0)

if elm_type != REGRESSION:
    print 'Not implemented yet!'


##################################################################
##################### CALCULATE WEIGHT AND BIAS ##################
t0 = time()

# Random generate input weight w_i and bias b_i of hidden neuron
input_weights =np.mat(np.random.rand(num_hidden_neuron, num_input_neuron) * 2 - 1)
bias_hidden_neuron = np.mat(np.random.rand(num_hidden_neuron, 1))
temp_H = np.mat(input_weights * P)
ind = np.mat(np.ones((1, num_train_data)))
bias_matrix = bias_hidden_neuron * ind # Extend the bias matrix to match the dimension of H
temp_H = temp_H + bias_matrix


##################################################################
############ CALCULATE HIDDEN NEURON OUTPUT MATRIX H #############
if activation_function == 'sigmoid':
    # equal to MATLAB code -> H = 1 ./ (1 + exp(-tempH));
    H = np.mat(np.divide(1, (1 + np.exp(np.multiply(-1, temp_H))))) # element wise divide and multiplication
elif activation_function == 'sine':
    H = np.mat(np.sin(temp_H))
elif activation_function == 'hardlim':
    H = utils.hardlim(temp_H)
elif activation_function == 'tribas':
    H = utils.triangular_bf(temp_H)
elif activation_function == 'radbas':
    H = utils.rad_bf(temp_H)
else:
    H = np.mat(np.divide(1, (1 + np.exp(np.multiply(-1, temp_H))))) # element wise divide and multiplication
    print 'Unknown Activation Function selected! Using default sigmoid as Activation Function instead...'

##################################################################
################ CALCULATE OUTPUT WEIGHTS beta_i #################
if pseudo_inverse_method == 'svd':
    output_weights = utils.pseudoinv_svd(H.T) * T.T
elif pseudo_inverse_method == 'geninv':
    output_weights = utils.pseudoinv_geninv(H)
elif pseudo_inverse_method == 'qrpivot':
    output_weights = utils.pseudoinv_qrpivot(H)
else:
    output_weights = utils.pseudoinv_svd(H.T) * T.T
    print 'Unknown Pseudo-Inverse method selected! Using default Moore-Penrose Pseudo-Inverse method instead...'

t1 = time()
train_time = t0 - t1 # time to train the ELM


##################################################################
################## CALCULATE TRAINING ACCURACY ###################
Y = np.mat(H.T * output_weights).T # Y: the actual output of the training data
Y = np.squeeze(np.asarray(Y)) # Squeeze matrix to one dimension array
# print np.squeeze(Y), x_sample
if elm_type == REGRESSION:
    train_accuracy = utils.compute_rmse(T, Y)
    print 'Train Accuracy = ' + str(train_accuracy)


fig = plt.figure('TRAIN REGRESSION')
ax = fig.add_subplot(111)
ax.plot(x, y_true, 'g-', linewidth=2, label='True')
ax.scatter(x_sample, y_sample1, s=50, facecolors='none', edgecolors='b', linewidths=0.5, label='Train Data1')
ax.scatter(x_sample, y_sample2, s=50, facecolors='none', edgecolors='r', linewidths=0.5, label='Train Data2')
ax.plot(x_sample, Y, 'y--', linewidth=2, label='Train Output')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(x_sample)))
plt.legend()
plt.grid()
# plt.show()

##################################################################
############### CALCULATE OUTPUT OF TESTING INPUT ################
t2 = time()
temp_H_test = input_weights * TVP
ind = np.mat(np.ones((1, num_test_data)))
bias_matrix = bias_hidden_neuron * ind # Extend the bias matrix to match the dimension of H
temp_H_test = temp_H_test + bias_matrix
if activation_function == 'sigmoid':
    # equal to MATLAB code -> H = 1 ./ (1 + exp(-tempH));
    H_test = np.mat(np.divide(1, (1 + np.exp(np.multiply(-1, temp_H_test))))) # element wise divide and multiplication
elif activation_function == 'sine':
    H_test = np.mat(np.sin(temp_H_test))
elif activation_function == 'hardlim':
    H_test = utils.hardlim(temp_H_test)
elif activation_function == 'tribas':
    H_test = utils.triangular_bf(temp_H_test)
elif activation_function == 'radbas':
    H_test = utils.rad_bf(temp_H_test)
else:
    H_test = np.mat(np.divide(1, (1 + np.exp(np.multiply(-1, temp_H_test))))) # element wise divide and multiplication
    print 'Unknown Activation Function selected! Using default sigmoid as Activation Function instead...'

TY = np.mat(H_test.T * output_weights).T # TY: the actual output of the testing data
t3 = time()
test_time = t3 - t2 # ELM time to predict the whole testing data
TY = np.squeeze(np.asarray(TY)) # Squeeze matrix to one dimension array
# print np.squeeze(Y), x_sample


##################################################################
################## CALCULATE TRAINING ACCURACY ###################
if elm_type == REGRESSION:
    test_accuracy = utils.compute_rmse(TVT, TY)
    print 'Test Accuracy = ' + str(test_accuracy)

fig = plt.figure('TEST REGRESSION')
ax = fig.add_subplot(111)
ax.plot(x, y_true, 'g-', linewidth=2, label='True')
ax.scatter(x_sample, y_test1, s=50, facecolors='none', edgecolors='b', linewidths=0.5, label='Test Data1')
ax.scatter(x_sample, y_test2, s=50, facecolors='none', edgecolors='r', linewidths=0.5, label='Test Data2')
ax.plot(x_sample, TY, 'y--', linewidth=2, label='Test Output')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with ELM with N data = ' + str(len(x_sample)))
plt.legend()
plt.grid()
plt.show()

if elm_type == CLASSIFIER:
    print 'Not implemented yet!'
