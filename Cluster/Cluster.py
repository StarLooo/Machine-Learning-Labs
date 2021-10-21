# -*- coding: UTF-8 -*-
import abc
import os
from abc import ABC

import numpy as np

from utils import *


class Cluster(metaclass=abc.ABCMeta):
    # 初始化
    def __init__(self, k, n_features, train_samples_matrix, clustered_samples_list=None, verbose=False):
        assert k > 0 and n_features > 0 and len(train_samples_matrix) > 0
        assert train_samples_matrix.shape[1] == n_features
        if clustered_samples_list is not None:
            self.ref_cluster_num = len(clustered_samples_list)
            assert len(clustered_samples_list) > 0
        self.k = k
        self.n_features = n_features
        self.n_samples = len(train_samples_matrix)
        self.train_samples_matrix = train_samples_matrix
        self.clustered_samples_list = clustered_samples_list
        self.verbose = verbose
        self.task_name = str(k) + " Cluster Task With " + str(self.n_samples) + " Samples."
        if self.verbose:
            print("Initial " + self.task_name)

    # 距离度量,计算source_point到target_point的距离
    def _distance(self, source_point, target_point, metric="Euclid"):
        assert source_point.shape == target_point.shape == (self.n_features,)
        if metric == "Euclid":
            return np.linalg.norm(source_point - target_point)
        else:
            raise NotImplementedError

    # 距离度量，计算point_matrix_A中各point到point_matrix_A中各point之间的距离的平方
    def _distances(self, point_matrix_A,point_matrix_B, metric="Euclid"):
        assert point_matrix.shape[1] == self.n_features
        if metric == "Euclid":
            return np.matmul(point_matrix, point_matrix.T)
        else:
            raise NotImplementedError

    # 计算轮廓系数
    def compute_silhouette_coefficient(self, samples, means_vectors, clusters):
        assert samples.shape == (self.n_samples, self.n_features)
        assert means_vectors.shape == (self.k, self.n_features)
        assert clusters.shape == (self.n_samples,)

    @abc.abstractmethod
    def cluster(self, draw_result=False):
        pass


class K_Means(Cluster, ABC):
    # 初始化
    def __init__(self, k, n_features, params, train_samples_matrix, clustered_samples_list=None,
                 init_method="heuristic", verbose=False):
        super().__init__(k, n_features, train_samples_matrix, clustered_samples_list, verbose)
        self.max_iters, self.epsilon = params
        self.init_method = init_method
        assert self.max_iters > 0 and self.epsilon > 0

    def __init_mean_vectors(self, init_method="heuristic"):
        if init_method == "heuristic":
            mean_vectors = []
            selected_samples = []
            remain_indexes = list(range(self.n_samples))
            np.random.shuffle(remain_indexes)
            selected_sample = self.train_samples_matrix[remain_indexes[0], :]
            mean_vectors.append(selected_sample)
            del remain_indexes[0]
            for i in range(self.k - 1):
                max_distance = -1
                choose_index = -1
                for index in remain_indexes:
                    distances = [self._distance(self.train_samples_matrix[index, :], selected_sample) for
                                 selected_sample in mean_vectors]
                    new_min_dist = np.min(distances)
                    if new_min_dist > max_distance:
                        max_distance = new_min_dist
                        choose_index = index
                remain_indexes.remove(choose_index)
                selected_sample = self.train_samples_matrix[choose_index, :]
                mean_vectors.append(selected_sample)
            mean_vectors = np.vstack(mean_vectors)
            assert mean_vectors.shape == (self.k, self.n_features)
            return mean_vectors
        elif init_method == "random":
            indexes = np.random.choice(a=self.n_samples, size=self.k)
            mean_vectors = self.train_samples_matrix[indexes, :]
            assert mean_vectors.shape == (self.k, self.n_features)
            return mean_vectors
        else:
            raise NotImplementedError

    def __mean_square_loss(self, samples, means_vectors, clusters):
        assert samples.shape == (self.n_samples, self.n_features)
        assert means_vectors.shape == (self.k, self.n_features)
        assert clusters.shape == (self.n_samples,)
        square_loss = 0
        for cluster_index in range(self.k):
            mask = (clusters == cluster_index)
            samples_in_this_cluster = self.train_samples_matrix[mask]
            square_loss += np.sum(np.linalg.norm(samples_in_this_cluster - means_vectors[cluster_index], axis=1) ** 2)
        mean_square_loss = square_loss / self.n_samples
        return mean_square_loss

    def __cluster_by_k_means(self, draw_result):
        pred_clusters = np.empty(self.n_samples)
        mean_square_loss = 1000000  # init with any val > epsilon
        means_vectors = self.__init_mean_vectors(self.init_method)
        self.init_means_vectors = means_vectors.copy()
        actual_iter_times = 0
        for iter_time in range(1, self.max_iters + 1):
            actual_iter_times = iter_time
            # 这里用向量化计算距离矩阵而不是循环，可以提高效率
            samples_inner_product = np.diag(np.matmul(self.train_samples_matrix, self.train_samples_matrix.T))
            assert samples_inner_product.shape == (self.n_samples,)

            means_inner_product = np.diag(np.matmul(means_vectors, means_vectors.T))
            assert means_inner_product.shape == (self.k,)

            samples_dist_matrix = samples_inner_product.reshape(self.n_samples, 1).repeat(repeats=self.k, axis=1)
            assert samples_dist_matrix.shape == (self.n_samples, self.k)

            means_dist_matrix = means_inner_product.reshape(1, self.k).repeat(repeats=self.n_samples, axis=0)
            assert means_dist_matrix.shape == (self.n_samples, self.k)

            cross_matrix = np.matmul(self.train_samples_matrix, means_vectors.T)
            assert cross_matrix.shape == (self.n_samples, self.k)
            dist_matrix = samples_dist_matrix - 2 * cross_matrix + means_dist_matrix
            pred_clusters = np.argmin(dist_matrix, axis=1)
            for cluster_index in range(self.k):
                mask = (pred_clusters == cluster_index)
                samples_in_this_cluster = self.train_samples_matrix[mask]
                means_vectors[cluster_index] = np.mean(samples_in_this_cluster, axis=0)
            pre_loss = mean_square_loss
            mean_square_loss = self.__mean_square_loss(self.train_samples_matrix, means_vectors, pred_clusters)
            if abs(pre_loss - mean_square_loss) <= self.epsilon:
                if self.verbose:
                    print("k-means algorithm has already been convergent, actual iter times:", actual_iter_times)
                    print("last mean_square_loss:", mean_square_loss)
                break
        if actual_iter_times == self.max_iters:
            if self.verbose:
                print("iter too many times, terminated!")
                print("last mean_square_loss:", mean_square_loss)

        if draw_result:
            assert 0 < k <= 6  # 为了画图颜色和图例的方便，最多支持6个簇
            marker_list = [".", "*", "^", "p", "+", "D"]
            color_list = ["tomato", "deepskyblue", "orange", "violet", "green", "teal"]
            for cluster_index in range(k):
                mask = (pred_clusters == cluster_index)
                samples_in_this_cluster = self.train_samples_matrix[mask]
                assert self.n_features == 2  # 为了方便和可行性，只实现了对二维特征的样本的可视化
                plt.scatter(samples_in_this_cluster[:, 0], samples_in_this_cluster[:, 1],
                            marker=marker_list[cluster_index],
                            color=color_list[cluster_index], s=20, label='cluster ' + str(cluster_index))
            plt.scatter(means_vectors[:, 0], means_vectors[:, 1],
                        marker="x", color="black", s=40, label="predict center")
            plt.scatter(self.init_means_vectors[:, 0], self.init_means_vectors[:, 1],
                        marker="x", color="grey", s=40, label="initial center")
            if self.clustered_samples_list is not None:
                generate_means_vectors = np.empty((self.k, self.n_features))
                for cluster_index in range(k):
                    generate_means_vectors[cluster_index] = np.mean(self.clustered_samples_list[cluster_index], axis=0)
                plt.scatter(generate_means_vectors[:, 0], generate_means_vectors[:, 1], marker="x", color="green", s=40,
                            label="generate center")
            title = "Predict Clustered Samples Graph(k=" + str(k) + ", init"
            if self.init_method == "random":
                title += " randomly)"
            elif self.init_method == "heuristic":
                title += " heuristically)"
            plt.title(title)
            plt.legend(loc="best")
            plt.xlabel("d_1")
            plt.ylabel("d_2")
            # plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
            plt.show()

        return pred_clusters, mean_square_loss, means_vectors, actual_iter_times

    def cluster(self, draw_result=False):
        if self.verbose:
            print("cluster using k-means:")
        return self.__cluster_by_k_means(draw_result)


if __name__ == '__main__':
    k = 3
    n_features = 2
    params = [100, 1e-12]
    mean_vector_list = [np.array([2, 4]), np.array([3, -6]), np.array([-2, - 4])]
    covariance_matrix_list = [np.array([[0.3, 0], [0, 0.6]]), np.array([[0.5, 0], [0, 0.5]]),
                              np.array([[0.55, 0], [0, 0.35]])]
    sample_num_list = [100, 100, 100]
    train_samples_matrix, clustered_samples_list = generate_data(k, n_features, sample_num_list, mean_vector_list,
                                                                 covariance_matrix_list)
    verbose = True

    k_means = K_Means(k, n_features, params, train_samples_matrix, clustered_samples_list, "random", verbose)
    pred_clusters, mean_square_loss, means_vectors, actual_iter_times = k_means.cluster(draw_result=True)

    k_means = K_Means(k, n_features, params, train_samples_matrix, clustered_samples_list, "heuristic", verbose)
    pred_clusters, mean_square_loss, means_vectors, actual_iter_times = k_means.cluster(draw_result=True)

# 向量化的sigmoid函数
# def _sigmoid(self, vector_x):
#     # 参考网上的经验，用这种方法解决溢出问题
#     # 把大于0和小于0的元素分别处理
#     # 当vector_x是比较小的负数时会出现上溢，此时可以通过计算exp(vector_x) / (1+exp(vector_x)) 来解决
#
#     mask = (vector_x > 0)
#     positive_out = np.zeros_like(vector_x, dtype='float64')
#     negative_out = np.zeros_like(vector_x, dtype='float64')
#
#     # 大于0的情况
#     positive_out = 1 / (1 + np.exp(-vector_x, positive_out, where=mask))
#     # 清除对小于等于0元素的影响
#     positive_out[~mask] = 0
#
#     # 小于等于0的情况
#     expZ = np.exp(vector_x, negative_out, where=~mask)
#     negative_out = expZ / (1 + expZ)
#     # 清除对大于0元素的影响
#     negative_out[mask] = 0
#
#     return positive_out + negative_out

# 预测
# def predict(self, w, X):
#     extend_X = np.concatenate((np.ones(shape=(len(X), 1)), X), axis=1)
#     y_pred = np.array([np.dot(w, extend_X[i, :]) for i in range(len(X))])
#     return y_pred

# 精度计算
# def _accuracy(self, w, X, y_true):
#     assert len(X) == len(y_true)
#     y_pred = (self.predict(w, X) > 0).astype(int)
#     accuracy = np.sum(y_pred == y_true.astype(np.int32)) / len(y_true)
#     assert 0.0 <= accuracy <= 1.0
#     return accuracy

# 优化目标
# def _optimize_objective_func(self, w, X, y):
#     assert len(X) == len(y)
#     extend_X = np.concatenate((np.ones(shape=(len(X), 1)), X), axis=1)
#     loss = 0
#     for i in range(len(X)):
#         inner_product = np.dot(w, extend_X[i, :])
#         # print("inner_product:", inner_product)
#         # 这种处理是防止浮点溢出
#         if inner_product > 15:
#             loss += -y[i] * inner_product + inner_product
#         else:
#             loss += -y[i] * inner_product + np.log1p(np.exp(inner_product))
#     regular = np.dot(w, w)
#     loss += 0.5 * self.regular_coef * regular
#     return loss

# 牛顿法求解
# def _solve_newton_method(self, max_iter_times: int, epsilon: float, draw_result: bool = False):
#     assert max_iter_times > 0 and epsilon >= 0.
#     # 初始化w
#     w = np.zeros(self.n_feature + 1)
#
#     # 牛顿法求解w
#     # 传入Newton_Optimizer的一阶导函数
#     def first_grad_func(w):
#         extend_X = np.concatenate((np.ones(shape=(self.n_train, 1)), self.X_train,), axis=1)
#         assert extend_X.shape == (self.n_train, self.n_feature + 1)
#         first_grad = np.matmul(extend_X.T,
#                                self._sigmoid(np.matmul(extend_X, w)) - self.y_train) + self.regular_coef * w
#         assert first_grad.shape == (self.n_feature + 1,)
#         return first_grad
#
#     # 传入Newton_Optimizer的二阶导函数
#     def second_grad_func(w):
#         extend_X = np.concatenate((np.ones(shape=(self.n_train, 1)), self.X_train), axis=1)
#         assert extend_X.shape == (self.n_train, self.n_feature + 1)
#         p1 = self._sigmoid(np.matmul(extend_X, w))
#         p0 = 1 - p1
#         p = p0 * p1
#         assert p.shape == p0.shape == p1.shape == (self.n_train,)
#         V = np.diag(p)
#         second_grad = np.matmul(np.matmul(extend_X.T, V), extend_X) + self.regular_coef * np.eye(self.n_feature + 1)
#         assert second_grad.shape == (self.n_feature + 1, self.n_feature + 1)
#         return second_grad
#
#     # 传入Gradient_Descent_Optimizer的train_loss计算函数
#     def loss_func(w):
#         loss = self._optimize_objective_func(w, self.X_train, self.y_train)
#         return loss
#
#     newton_opt = Newton_Optimizer(w, [max_iter_times, epsilon], loss_func, first_grad_func, second_grad_func,
#                                   self.verbose)  # 初始化Newton_Optimizer
#     w, actual_iter_times, train_loss_list, first_grad, second_grad = newton_opt.train()  # 使用牛顿法求出w的解
#
#     # 计算训练完成后训练集测试集上的accuracy
#     train_acc = self._accuracy(w, self.X_train, self.y_train)
#     test_acc = self._accuracy(w, self.X_test, self.y_test)
#
#     if self.verbose:
#         print("L1-norm of latest first gradient:", np.max(first_grad))
#         print("w:", w)
#         print("train_acc:", round(train_acc, 4), "test_acc:", round(test_acc, 4))
#         print("actual iter times:", actual_iter_times)
#
#     # 画图分析结果
#     if draw_result:
#         title = "newton method"
#         if self.regular_coef > 0.0:
#             title += ", regular_coef=" + str(round(self.regular_coef, 4))
#         draw_predict_analysis(self.X_train, self.train_pos_samples_num, self.X_test, self.test_pos_samples_num, w,
#                               title)
#
#     return w, train_acc, test_acc, actual_iter_times, train_loss_list

# 梯度下降法求解
# def _solve_gradient_descent(self, lr: float, max_iter_times: int, epsilon: float, draw_result: bool = False):
#     assert lr > 0., max_iter_times > 0 and epsilon >= 0.
#     # 初始化w
#     w = np.zeros(self.n_feature + 1)
#
#     # 梯度下降法求解w
#     # 传入Gradient_Descent_Optimizer的梯度函数
#     def grad_func(w):
#         extend_X = np.concatenate((np.ones(shape=(self.n_train, 1)), self.X_train), axis=1)
#         assert extend_X.shape == (self.n_train, self.n_feature + 1)
#         # 以下累加式可以向量化，向量化后numpy可以大大提高计算效率
#         # grad = np.zeros(self.n_feature + 1)
#         # for i in range(self.n_train):
#         #     # 注意防止浮点溢出
#         #     inner_product = np.dot(w, extend_X[i, :])
#         #     if inner_product >= 0:
#         #         power = np.exp(-inner_product)
#         #         grad -= extend_X[i, :] * (self.y_train[i] - 1 + power / (1 + power))
#         #     else:
#         #         grad -= extend_X[i, :] * (self.y_train[i] - 1 + 1 / (1 + np.exp(inner_product)))
#         # grad += self.regular_coef * w
#         grad = np.matmul(extend_X.T, self._sigmoid(np.matmul(extend_X, w)) - self.y_train) + self.regular_coef * w
#         assert grad.shape == (self.n_feature + 1,)
#         return grad
#
#     # 传入Gradient_Descent_Optimizer的train_loss计算函数
#     def loss_func(w):
#         loss = self._optimize_objective_func(w, self.X_train, self.y_train)
#         return loss
#
#     gd_opt = Gradient_Descent_Optimizer(w, [lr, max_iter_times, epsilon], loss_func, grad_func,
#                                         self.verbose)  # 初始化Gradient_Descent_Optimizer
#     w, actual_iter_times, train_loss_list, latest_grad = gd_opt.train()  # 使用梯度下降法求出w的解
#
#     # 计算训练完成后训练集测试集上的accuracy
#     train_acc = self._accuracy(w, self.X_train, self.y_train)
#     test_acc = self._accuracy(w, self.X_test, self.y_test)
#
#     if self.verbose:
#         print("L1-norm of latest gradient:", np.max(np.abs(latest_grad)))
#         print("w:", w)
#         print("train_acc:", round(train_acc, 4), "test_acc:", round(test_acc, 4))
#         print("actual iter times:", actual_iter_times)
#
#     # 画图分析结果
#     if draw_result:
#         title = "gradient descent"
#         if self.regular_coef > 0.0:
#             title += ", regular_coef=" + str(round(self.regular_coef, 4))
#         draw_predict_analysis(self.X_train, self.train_pos_samples_num, self.X_test, self.test_pos_samples_num, w,
#                               title)
#
#     return w, train_acc, test_acc, actual_iter_times, train_loss_list

# 训练
