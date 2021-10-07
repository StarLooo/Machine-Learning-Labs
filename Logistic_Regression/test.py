import numpy as np

if __name__ == '__main__':
    a = np.array([1, 2, -1]).reshape(3,1)
    b = np.array([[1, 2], [3, 4], [5, 6]])
    print(a * b)
    X = np.array([[1, -1], [2, -2]])
    print((X > 0).astype(int))
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    tail = np.array([[12, 9, 8, 10]])
    A = np.concatenate((X, tail.T), axis=1)
    print(A)
