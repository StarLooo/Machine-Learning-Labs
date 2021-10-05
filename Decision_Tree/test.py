import os

import numpy as np

from Polynomial_Regression.utils import *

if __name__ == '__main__':
    a = np.array([1,2,3])
    b = np.array([5,6,7])
    print(a.dot(b))
    os.system("pause")
    x = np.array([2,3,4])
    n = 3
    m = 3
    X = np.vander(x, m + 1, increasing=True)
    X_t = X.transpose()
    print(X, "\n", X_t)
    X_t_mul_X = X_t @ X
    print(X_t_mul_X)
    X_t_mul_X_inv = np.linalg.inv(X_t_mul_X)
    X_t_mul_X_pinv = np.linalg.pinv(X_t_mul_X)
    print("X_t_mul_X_inv:")
    print(X_t_mul_X_inv)
    print("np.matmul(X_t_mul_X_inv, X_t_mul_X):")
    print(np.matmul(X_t_mul_X_inv, X_t_mul_X))
    print("X_t_mul_X_pinv:")
    print(X_t_mul_X_pinv)
    print("np.matmul(np.matmul(X_t_mul_X, X_t_mul_X_pinv), X_t_mul_X):")
    print(np.matmul(np.matmul(X_t_mul_X, X_t_mul_X_pinv), X_t_mul_X))
    os.system("pause")
    indexes = {0, 1, 3, 7}
    X = np.array([8, 1, 2, 1, 4, 15, 1, 7])
    Y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    print(np.argmax(X))
    print((np.sum(np.square(X - Y))) / 2)
    # print(set(X))
    # print(type(np.array((1, 2))))
    # print(type("AVS"))
