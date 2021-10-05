import numpy as np

if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    tail = np.array([[12, 9, 8, 10]])
    A = np.concatenate((X, tail.T), axis=1)
    print(A)
