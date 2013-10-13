__author__ = 'iwawiwi'


import numpy as np


def hardlim(x):
    """
    Hard Limit transfer function

    MATLAB algorithm
        hardlim(n) =    1 if n >= 0
                        0 otherwise
    """
    # x = np.mat([[1,2,3,-1,2,-3],[-1,2,-3,1,2,-3]])
    # print x, np.size(x, 0) - 1, np.size(x, 1) - 1
    for i in range(0, np.size(x, 0)):
        for j in range(0, np.size(x, 1)):
            # print i, j
            if x[i,j] >= 0:
                x[i,j] = 1
            else:
                x[i,j] = 0
    # print x
    return np.mat(x, dtype='f') # cast matrix x as float data type


def triangular_bf(x):
    """
    Triangular Basis Function

    MATLAB algorithm
        a = tribas(n)   = 1 - abs(n), if -1 <= n <= 1
                        = 0, otherwise
    """
    # x = np.mat([[1,2,3,-1,2,-3],[-1,2,-3,1,2,-3]])
    for i in range(0, np.size(x, 0)):
        for j in range(0, np.size(x, 1)):
            # print i, j
            if -1 <= x[i,j] <= 1:
                x[i,j] = 1 - np.abs(x[i,j])
            else:
                x[i,j] = 0
    # print x
    return x


def rad_bf(x):
    """
    Radian Basis Function

    MATLAB algorithm
        a = radbas(n) = exp(-n^2)
    """
    # x = np.mat([[1,2,3,-1,2,-3],[-1,2,-3,1,2,-3]], dtype='f')
    # x = np.mat([[0.2123,-0.123,-0.3425,0.1343],[-0.5123,-0.623,0.5425,0.343]])
    for i in range(0, np.size(x, 0)):
        for j in range(0, np.size(x, 1)):
            # print i, j
            # print np.dtype(np.exp(-pow(x[i,j], 2)))
            x[i,j] = np.exp(-pow(x[i,j], 2))
    # print np.array(x, dtype='f')
    return x


def pseudoinv_svd(x):
    """
    Moore-Penrose Pseudo-Inverse Matrix
    """
    p_inv = np.linalg.pinv(x) # TODO: Temporary, using pinv numpy.linalg function
    return p_inv


# TODO: Not implemented yet!
def pseudoinv_geninv(x):
    """
    GENINV Pseudo-Inverse
    """
    return x


# TODO: Not implemented yet!
def pseudoinv_qrpivot(x):
    """
    QR-Pivot Pseudo-Inverse
    """
    return x


def compute_rmse(actual, computed):
    """
    Compute Root Mean Square Error
    """
    n = actual.size
    err = actual - computed
    sq_err = np.power(err, 2)
    sq_err_sum = sq_err.sum()
    mse = (1.0 / (2*n)) * sq_err_sum # Mean Square Error
    return np.sqrt(mse) # Return Root Mean Square Error