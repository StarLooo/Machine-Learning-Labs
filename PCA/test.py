import numpy as np

if __name__ == '__main__':
    X = np.array([[1, 5, 9], [7, 2, 3], [4, 8, 6], [1, 2, 3]])
    print(X[1])
    print(X[:, [1]])
    print(X - X.mean(axis=0))
    print(X - X.mean(axis=1))
