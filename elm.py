__author__ = 'iwawiwi'


def pseudoInverse():
    return

"""
ELM (Extreme Learning Machine)
taken from http://www.ntu.edu.sg/home/egbhuang/elm_random_hidden_nodes.html
"""
def elm(train_data, test_data, elm_type, num_hid_neuron, activation_func):
    # MACRO Definition
    REGRESSION = 0;
    CLASSIFIER = 1;

    # LOAD training dataset
    T = train_data[:,1]
    P = train_data[:,2] # TODO: Assume that train_data only consist of two column

    # LOAD testing dataset
    TVT = test_data[:,1]
    TVP = test_data[:,2] # TODO: Assume that test_data only consist of two column

    num_train_data = np.size(P,2)
    num_test_data = np.size(TVP,2)
    num_in_neuron = np.sizr(P,1)


    if elm_type ~= REGRESSION:
        # CLASSIFICATION rule

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
        H = np.mat(1 ./ (1 + np.exp(-tempH)))
    if activation_func == 'sine':
        H = np.mat(np.sin(tempH))

    # CALCULATE hidden neuron output matrix H
    out_weight = pseudoInverse(H.T) * T.T # do a pseudo inverse, implementation without regularization factor

    t1 = time()
    training_time = t1 - t0

    # CALCULATE training accuracy
    Y = np.mat((H.T * out_weight)).T
    if elm_type == REGRESSION:
        train_acc = computeRMSE(T,Y)

    # CALCULATE output of testing input
    t2 = time()
    tempH_test = in_weight * TVP
    ind = np.ones(1, num_test_data)
    bias_matrix = bias_hid_neuron[:, ind]
    tempH_test = tempH_test + bias_matrix

    if activation_func == 'sigmoid':
        H_test = np.mat(1 ./ (1 + np.exp(-tempH_test)))
    if activation_func == 'sine':
        H_test = np.mat(np.sin(tempH_test))

    TY = np.mat((H_test.T * out_weight)).T
    t3 = time()
    test_time = t3 - t2

    if elm_type == REGRESSION:
        test_acc = computeRMSE(TVT, TY)
    if elm_type == CLASSIFIER:
        # CLASSIFICATION rule

    return