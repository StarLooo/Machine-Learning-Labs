# -*- coding: UTF-8 -*-
import os

import numpy as np

if __name__ == '__main__':
    a = np.array([1, 3, 2, 4, 6])
    b = np.array([2, 1, 1, 5, 7])
    c = np.array([2,3,2,5,7])
    print(c / np.maximum(a, b))
    os.system("pause")

    v = np.array([1, 2, 3, 4, 5])
    a = v.reshape(5, -1).repeat(repeats=3, axis=1)
    b = v.reshape(1, -1).repeat(repeats=3, axis=0)
    print(a)
    print(b)

    M = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
    print(np.sum(M,axis=1))
    os.system("pause")
    indexes = np.random.choice(a=7, size=3)
    print(indexes)
    m = M[indexes, :]

    print("M")
    M = np.array([[2, 1, 3], [6, 5, 4]])  # (2,3)
    print(np.argmin(M, axis=1))

    v = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    print(b[v == 1])

    M = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]])
    print(np.mean(M, axis=0))
    print(np.mean(M, axis=1))

    MM = M - np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    print(MM)
    print(np.linalg.norm(MM, axis=1) ** 2)

