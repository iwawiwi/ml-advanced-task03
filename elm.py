__author__ = 'iwawiwi'


import numpy as np
from time import time

def compute_rmse(x,y):
    rmse = 0
    return rmse


def pseudo_inverse_svdmp(x):
    return x


def pseudo_inverse_geninv(x):
    return x


def pseudo_inverse_qrpivot(x):
    return x

def elm(train_data, test_data, elm_type, num_hid_neuron, activation_func, pseudo_inv_param):
    """
    ELM (Extreme Learning Machine)
    taken from http://www.ntu.edu.sg/home/egbhuang/elm_random_hidden_nodes.html1
    """
    # MACRO Definition
    REGRESSION = 0
    CLASSIFIER = 1

    # LOAD training dataset
    T = train_data[:,1]
    P = train_data[:,2] # TODO: Assume that train_data only consist of two column

    # LOAD testing dataset
    TVT = test_data[:,1]
    TVP = test_data[:,2] # TODO: Assume that test_data only consist of two column

    num_train_data = np.size(P,2)
    num_test_data = np.size(TVP,2)
    num_in_neuron = np.size(P,1)


    if elm_type != REGRESSION:
        # CLASSIFICATION rule
        print 'classification rule'

    ################################ START TRAINING #################################
    t0 = time() # start the time
    # Random generate input weight w_i and biases b_i of hidden neuron
    in_weight = np.random.rand(num_hid_neuron, num_in_neuron) * 2 - 1
    bias_hid_neuron = np.random.rand(num_hid_neuron, 1)
    tempH = in_weight * P

    ind = np.ones(1, num_train_data)
    bias_matrix = bias_hid_neuron[:,ind] # Extend the bias matrix to match the demention of H
    tempH = tempH + bias_matrix

    # activation funciton
    if activation_func == 'sigmoid':
        H = np.mat(1 / (1 + np.exp(-tempH)))
    elif activation_func == 'sine':
        H = np.mat(np.sin(tempH))
    elif activation_func == 'hardlim':
        H = np.mat(tempH)
    else:
        H = np.mat(np.ones(shape=(1,1)))

    # CALCULATE hidden neuron output matrix H
    if pseudo_inv_param == 'SVDMP':
        out_weight = pseudo_inverse_svdmp(H.T) * T.T # do a pseudo inverse, implementation without regularization factor
    elif pseudo_inv_param == 'GENINV':
        out_weight = pseudo_inverse_geninv(H.T)
    else:
        out_weight = pseudo_inverse_qrpivot(H.T)

    t1 = time()
    training_time = t1 - t0

    # CALCULATE training accuracy
    Y = np.mat((H.T * out_weight)).T
    if elm_type == REGRESSION:
        train_acc = compute_rmse(T,Y)

    # CALCULATE output of testing input
    t2 = time()
    tempH_test = in_weight * TVP
    ind = np.ones(1, num_test_data)
    bias_matrix = bias_hid_neuron[:, ind]
    tempH_test = tempH_test + bias_matrix

    if activation_func == 'sigmoid':
        H_test = np.mat(1 / (1 + np.exp(-tempH_test)))
    if activation_func == 'sine':
        H_test = np.mat(np.sin(tempH_test))
    else:
        H_test = np.mat(np.ones(shape=(1,1)))

    TY = np.mat((H_test.T * out_weight)).T
    t3 = time()
    test_time = t3 - t2

    if elm_type == REGRESSION:
        test_acc = compute_rmse(TVT, TY)
    if elm_type == CLASSIFIER:
        # CLASSIFICATION rule
        print 'classifier rule'

    return