__author__ = 'iwawiwi'


import numpy as np
from time import time
import utils

def elm(train_data, test_data, elm_type, num_hidden_neuron, activation_function, pseudo_inverse_method):
    """
    ELM taken from http://www.ntu.edu.sg/home/egbhuang/elm_random_hidden_nodes.html
    """
    REGRESSION = 0
    CLASSIFIER = 1
    ##################################################################
    ######################## LOAD TRAINING DATA SET ##################
    T = np.mat(train_data[:,0].T)
    P = np.mat(train_data[:,1:np.size(train_data,1)].T)
    #print 'T: ', T.shape
    #print 'P: ', P.shape


    ##################################################################
    ######################## LOAD TESTING DATA SET ###################
    TVT = np.mat(test_data[:,0].T)
    TVP = np.mat(test_data[:,1:np.size(test_data,1)].T)

    # Initialize NUMBER of NEURON, TEST DATA, and TRAIN DATA
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
        output_weights = utils.pseudoinv_geninv(H.T) * T.T
    elif pseudo_inverse_method == 'qrpivot':
        output_weights = utils.pseudoinv_qrpivot(H.T) * T.T
    else:
        output_weights = utils.pseudoinv_svd(H.T) * T.T
        print 'Unknown Pseudo-Inverse method selected! Using default Moore-Penrose Pseudo-Inverse method instead...'

    t1 = time()
    train_time = t1 - t0 # time to train the ELM
    print 'Train Time = ' + str(train_time)


    ##################################################################
    ################## CALCULATE TRAINING ACCURACY ###################
    Y = np.mat(H.T * output_weights).T # Y: the actual output of the training data
    print 'Y_ELM: ', Y
    Y = np.squeeze(np.asarray(Y)) # Squeeze matrix to one dimension array
    # print np.squeeze(Y), x_sample
    train_accuracy = 0
    if elm_type == REGRESSION:
        train_accuracy = utils.compute_rmse(T, Y)
        print 'Train Accuracy = ' + str(train_accuracy)

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
    print 'TY_ELM: ', TY
    t3 = time()
    test_time = t3 - t2 # ELM time to predict the whole testing data
    print 'Test Time = ' + str(test_time)
    TY = np.squeeze(np.asarray(TY)) # Squeeze matrix to one dimension array
    # print np.squeeze(Y), x_sample


    ##################################################################
    ################## CALCULATE TRAINING ACCURACY ###################
    test_accuracy = 0
    if elm_type == REGRESSION:
        test_accuracy = utils.compute_rmse(TVT, TY)
        print 'Test Accuracy = ' + str(test_accuracy)

    if elm_type == CLASSIFIER:
        print 'Not implemented yet!'

    return Y, TY, train_accuracy, test_accuracy