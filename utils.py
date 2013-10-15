__author__ = 'iwawiwi'


import numpy as np
from time import time


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
    t0 = time()
    p_inv = np.linalg.pinv(x) # TODO: Still using pinv numpy.linalg function
    t1 = time()
    elapsed_time = t1 - t0
    print 'Elapsed time for computing Pseudo-Inverse using PINV method: ', str(elapsed_time)
    return p_inv


def pseudoinv_geninv(x):
    """
    GENINV Pseudo-Inverse
    Taken from MATLAB source code of original paper
    """
    t0 = time()
    x = np.mat(x)
    m = np.size(x,0)
    n = np.size(x,1)
    transpose = False
    if m < n:
        #print 'transpose'
        transpose = True
        A = x * x.T
    else:
        A = x.T * x
    #print 'A: ', A

    # Full rank cholesky factorization
    L = np.linalg.cholesky(A) # TODO: Still using cholesky numpy.linalg function
    #print 'L: ', L
    LTLI = (L.T * L).I
    #print 'LTLI: ', LTLI

    pseudo_inv = L * LTLI * LTLI * L.T
    if transpose:
        #print 'transpose'
        geninv = x.T * pseudo_inv
    else:
        #print 'pseudo_inv: ', pseudo_inv
        #print 'x.T: ', x.T
        geninv = pseudo_inv * x.T
    t1 = time()
    elapsed_time = t1 - t0
    print 'Elapsed time for computing Pseudo-Inverse using GENINV method: ', str(elapsed_time)
    return geninv


# TODO: Not implemented yet!
def pseudoinv_qrpivot(x):
    """
    QR-Pivot Pseudo-Inverse
    """
    t0 = time()
    x = np.mat(x)
    m = np.size(x,0)
    n = np.size(x,1)
    if m >= n:
        r = np.linalg.qr(x, mode='r') # For fast computation TODO: Still using QR numpy.linalg function
        r = np.mat(r)
        # Full rank R
        # if np.linalg.matrix_rank(r) == n:
        qr_inv = (r.T * r).I * x.T
        t1 = time()
        elapsed_time = t1 - t0
        print 'Elapsed time for computing Pseudo-Inverse using QR-PIVOT method: ', str(elapsed_time)
        return qr_inv
    else:
        print 'Cannot decompose Matrix with QR-PIVOT!'
        raise Exception("Can't do QR-Pivot!")


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


# TODO: Testing purpose only
#A = np.mat([[1,2],[2,1],[2,3]])
#A = np.random.rand(500000,10)
#A = np.random.rand(3, 2)
#pse_inv = pseudoinv_svd(A)
#gen_inv = pseudoinv_geninv(A)
#qrpivot_inv = pseudoinv_qrpivot(A)
#
#print 'MP Pseudo-Inverse: ', pse_inv
#print 'GEN-INV Pseudo-Inverse', gen_inv
#print 'QR-PIVOT Pseudo-Inverse', qrpivot_inv