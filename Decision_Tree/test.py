import numpy as np

if __name__ == '__main__':
    indexes = {0, 1, 3, 7}
    X = np.array([8, 1, 2, 1, 4, 15, 1, 7])
    Y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    print(np.argmax(X))
    print((np.sum(np.square(X - Y))) / 2)
    # print(set(X))
    # print(type(np.array((1, 2))))
    # print(type("AVS"))
