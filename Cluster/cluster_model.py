# -*- coding: UTF-8 -*-
import abc
import itertools
from abc import ABC

import numpy as np
from scipy.stats import multivariate_normal

from utils import *


# 聚类模型的抽象接口
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
    def _distances(self, point_matrix_A, point_matrix_B, metric="Euclid"):
        assert point_matrix_A.shape[1] == point_matrix_B.shape[1] == self.n_features
        n_A = point_matrix_A.shape[0]
        n_B = point_matrix_B.shape[0]
        if metric == "Euclid":
            A_mul_A_T = np.diag(np.matmul(point_matrix_A, point_matrix_A.T))
            A_mul_A_T_repeat = A_mul_A_T.reshape(n_A, 1).repeat(repeats=n_B, axis=1)
            B_mul_B_T = np.diag(np.matmul(point_matrix_B, point_matrix_B.T))
            B_mul_B_T_repeat = B_mul_B_T.reshape(1, n_B).repeat(repeats=n_A, axis=0)
            cross_matrix = np.matmul(point_matrix_A, point_matrix_B.T)
            assert A_mul_A_T_repeat.shape == B_mul_B_T_repeat.shape == cross_matrix.shape == (n_A, n_B)
            return A_mul_A_T_repeat - 2 * cross_matrix + B_mul_B_T_repeat
        else:
            raise NotImplementedError

    # 计算轮廓系数
    def compute_silhouette_coefficient(self, pred_clusters):
        assert pred_clusters.shape == (self.n_samples,)
        samples_distances_matrix = self._distances(self.train_samples_matrix, self.train_samples_matrix) ** 0.5
        total_sc = 0
        for cluster_index in range(self.k):
            mask_inner = (pred_clusters == cluster_index)
            samples_distances_matrix_in_this_cluster = samples_distances_matrix[mask_inner, :][:, mask_inner]
            num_in_this_cluster = samples_distances_matrix_in_this_cluster.shape[1]
            if num_in_this_cluster > 1:
                a = np.sum(samples_distances_matrix_in_this_cluster, axis=1) / (num_in_this_cluster - 1)
            else:
                a = [0 for _ in range(num_in_this_cluster)]
            b = [1000000 for _ in range(num_in_this_cluster)]
            for other_cluster_index in range(self.k):
                if other_cluster_index != cluster_index:
                    mask_other = (pred_clusters == other_cluster_index)
                    samples_distances_matrix_to_another_cluster = \
                        samples_distances_matrix[mask_inner, :][:, mask_other]
                    num_in_another_cluster = samples_distances_matrix_to_another_cluster.shape[1]
                    if num_in_another_cluster > 0:
                        b = np.minimum(b, np.mean(samples_distances_matrix_to_another_cluster, axis=1))
                    else:
                        b = [0 for _ in range(num_in_another_cluster)]
            assert len(a) == len(b) == num_in_this_cluster
            total_sc += np.sum((b - a) / np.maximum(a, b))
        mean_sc = total_sc / self.n_samples
        assert -1.0 <= mean_sc <= 1.0
        return mean_sc

    # 计算精度(仅仅当有参考簇划分并且k的选取与参考一致时才有效)
    def compute_accuracy(self, pred_clusters, ref_clusters):
        assert self.clustered_samples_list is not None and len(self.clustered_samples_list) == self.k
        assert pred_clusters.shape == ref_clusters.shape == (self.n_samples,)
        accuracy = 0.0
        correct_num = 0
        aligned_clusters = np.empty_like(pred_clusters)
        for align_relation in itertools.permutations(range(self.k)):
            for cluster_index in range(self.k):
                aligned_clusters[pred_clusters == cluster_index] = align_relation[cluster_index]
            accuracy = max(accuracy, np.mean(aligned_clusters == ref_clusters))
        assert 0.0 <= accuracy <= 1.0
        return accuracy

    # 进行聚类
    @abc.abstractmethod
    def cluster(self, draw_result=False):
        pass


class K_Means(Cluster, ABC):
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

    def __mean_square_loss(self, samples, mean_vectors, clusters):
        assert samples.shape == (self.n_samples, self.n_features)
        assert mean_vectors.shape == (self.k, self.n_features)
        assert clusters.shape == (self.n_samples,)
        square_loss = 0
        for cluster_index in range(self.k):
            mask = (clusters == cluster_index)
            samples_in_this_cluster = self.train_samples_matrix[mask]
            square_loss += np.sum(np.linalg.norm(samples_in_this_cluster - mean_vectors[cluster_index], axis=1) ** 2)
        mean_square_loss = square_loss / self.n_samples
        return mean_square_loss

    def __cluster_by_k_means(self, draw_result):
        pred_clusters = np.empty(self.n_samples)
        mean_square_loss = 1000000  # init with any val > epsilon
        mean_vectors = self.__init_mean_vectors(self.init_method)
        self.init_mean_vectors = mean_vectors.copy()
        actual_iter_times = 0
        for iter_time in range(1, self.max_iters + 1):
            actual_iter_times = iter_time
            # 这里调用_distances()向量化计算距离矩阵而不是循环，可以提高效率
            dist_matrix = self._distances(self.train_samples_matrix, mean_vectors)
            pred_clusters = np.argmin(dist_matrix, axis=1)
            for cluster_index in range(self.k):
                mask = (pred_clusters == cluster_index)
                samples_in_this_cluster = self.train_samples_matrix[mask]
                mean_vectors[cluster_index] = np.mean(samples_in_this_cluster, axis=0)
            pre_loss = mean_square_loss
            mean_square_loss = self.__mean_square_loss(self.train_samples_matrix, mean_vectors, pred_clusters)
            if abs(pre_loss - mean_square_loss) <= self.epsilon:
                if self.verbose:
                    print("k-means algorithm has already been convergent, actual iter times:", actual_iter_times)
                break
            if actual_iter_times == self.max_iters:
                if self.verbose:
                    print("iter too many times, terminated!")
        if self.verbose:
            print("last mean_square_loss:", mean_square_loss)
            print("silhouette_coefficient:", self.compute_silhouette_coefficient(pred_clusters))

        if draw_result:
            assert 0 < self.k <= 6  # 为了画图颜色和图例的方便，最多支持6个簇
            assert self.n_features == 2  # 为了画图的方便，只支持二维特征
            marker_list = [".", "*", "^", "p", "+", "D"]
            color_list = ["tomato", "deepskyblue", "orange", "violet", "green", "teal"]
            for cluster_index in range(self.k):
                mask = (pred_clusters == cluster_index)
                samples_in_this_cluster = self.train_samples_matrix[mask]
                assert self.n_features == 2  # 为了方便和可行性，只实现了对二维特征的样本的可视化
                plt.scatter(samples_in_this_cluster[:, 0], samples_in_this_cluster[:, 1],
                            marker=marker_list[cluster_index],
                            color=color_list[cluster_index], s=20, label='cluster ' + str(cluster_index))
            plt.scatter(mean_vectors[:, 0], mean_vectors[:, 1],
                        marker="o", color="black", s=40, label="predict center")
            plt.scatter(self.init_mean_vectors[:, 0], self.init_mean_vectors[:, 1],
                        marker="x", color="grey", s=40, label="initial center")
            if self.clustered_samples_list is not None:
                actual_k = len(self.clustered_samples_list)
                generate_mean_vectors = np.empty((actual_k, self.n_features))
                for cluster_index in range(actual_k):
                    generate_mean_vectors[cluster_index] = np.mean(self.clustered_samples_list[cluster_index], axis=0)
                plt.scatter(generate_mean_vectors[:, 0], generate_mean_vectors[:, 1], marker="x", color="green", s=40,
                            label="generate center")
            title = "K-Means Clustered Samples Graph(k=" + str(self.k) + ", init"
            if self.init_method == "random":
                title += " randomly)"
            elif self.init_method == "heuristic":
                title += " heuristically)"
            plt.title(title)
            plt.legend(loc="best")
            plt.xlabel("d_1")
            plt.ylabel("d_2")
            plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
            plt.show()

        return pred_clusters, mean_square_loss, mean_vectors, actual_iter_times

    def cluster(self, draw_result=False):
        if self.verbose:
            print("clustering using k-means:")
        return self.__cluster_by_k_means(draw_result)


class Gaussian_Mixture_Model(Cluster, ABC):
    def __init__(self, k, n_features, params, train_samples_matrix, clustered_samples_list=None,
                 init_method="heuristic", verbose=False):
        super().__init__(k, n_features, train_samples_matrix, clustered_samples_list, verbose)
        self.max_iters, self.epsilon = params
        self.init_method = init_method
        assert self.max_iters > 0 and self.epsilon > 0

    def __init_alphas(self):
        return np.array([1 / self.k for _ in range(self.k)])

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

    def __init_covariance_matrixes(self):
        cov = 0.1 * np.eye(N=self.n_features)
        assert cov.shape == (self.n_features, self.n_features)
        return np.array([cov for _ in range(self.k)])

    def __compute_likelihood(self, mean_vectors, covariance_matrixes):
        assert mean_vectors.shape == (self.k, self.n_features) and covariance_matrixes.shape == (
            self.k, self.n_features, self.n_features)
        # P[j, i]表示在第i个高斯混合分布下生成样本xj的似然
        P = np.empty((self.n_samples, self.k))
        for col_index in range(self.k):
            mean_vector = mean_vectors[col_index]
            covariance_matrix = covariance_matrixes[col_index]
            P[:, col_index] = multivariate_normal.pdf(x=self.train_samples_matrix, mean=mean_vector,
                                                      cov=covariance_matrix, allow_singular=True)
        return P

    def __compute_posterior_probability(self, P, alphas):
        assert P.shape == (self.n_samples, self.k) and alphas.shape == (self.k,)
        # G[j, i]表示样本xj由第i个高斯混合分布生成的后验概率
        G = alphas.reshape(1, self.k) * P
        G = G / np.sum(G, axis=1, keepdims=True)
        assert G.shape == (self.n_samples, self.k)
        assert np.allclose(np.sum(G, axis=1), np.ones(self.n_samples))
        return G

    def __cluster_by_gmm(self, draw_result):
        pred_clusters = np.empty(self.n_samples)
        mean_vectors = self.__init_mean_vectors(self.init_method)
        self.init_mean_vectors = mean_vectors.copy()
        alphas = self.__init_alphas()
        covariance_matrixes = self.__init_covariance_matrixes()
        actual_iter_times = 0
        for iter_time in range(1, self.max_iters + 1):
            actual_iter_times = iter_time
            # E step:
            P = self.__compute_likelihood(mean_vectors, covariance_matrixes)
            G = self.__compute_posterior_probability(P, alphas)
            # M step:
            pre_mean_vectors = mean_vectors
            pre_covariance_matrixes = covariance_matrixes
            pre_alphas = alphas
            # update mean_vectors
            mean_vectors = np.matmul(G.T, self.train_samples_matrix) / np.sum(G, axis=0).reshape(self.k, 1)
            # TODO:这里的向量化遇到了困难，尤其是协方差矩阵更新的向量化
            for i in range(self.k):
                D = self.train_samples_matrix - mean_vectors[i]  # 注意mean_vectors[i]广播
                g = G[:, i]
                covariance_matrixes[i] = np.matmul(g.reshape(1, self.n_samples) * D.T, D) / np.sum(g)  # 注意这里的"*"是哈达玛积
            # update alphas
            alphas = np.mean(G, axis=0)
            norm_delta_mean_vectors = np.linalg.norm(mean_vectors - pre_mean_vectors)
            norm_delta_covariance_matrixes = np.linalg.norm(covariance_matrixes - pre_covariance_matrixes)
            norm_delta_alphas = np.linalg.norm(alphas - pre_alphas)
            if norm_delta_mean_vectors < self.epsilon and norm_delta_covariance_matrixes < self.epsilon and \
                    norm_delta_alphas < self.epsilon:
                if self.verbose:
                    print("gmm has already been convergent, actual iter times:", actual_iter_times)
                break
            if actual_iter_times == self.max_iters:
                if self.verbose:
                    print("iter too many times, terminated!")

        P = self.__compute_likelihood(mean_vectors, covariance_matrixes)
        G = self.__compute_posterior_probability(P, alphas)
        pred_clusters = np.argmax(G, axis=1)
        if self.verbose:
            print("silhouette_coefficient:", self.compute_silhouette_coefficient(pred_clusters))
            for i in range(self.k):
                print("--------------------parameters predict-------------------")
                print("alpha " + str(i) + ":", alphas[i])
                print("mean_vector " + str(i) + ":", mean_vectors[i])
                print("covariance_matrix " + str(i) + ":", "\n", covariance_matrixes[i])
                print("---------------------------------------------------------")

        if draw_result:
            assert 0 < self.k <= 6  # 为了画图颜色和图例的方便，最多支持6个簇
            assert self.n_features == 2  # 为了画图的方便，只支持二维特征
            marker_list = [".", "*", "^", "p", "+", "D"]
            color_list = ["tomato", "deepskyblue", "orange", "violet", "green", "teal"]
            for cluster_index in range(self.k):
                mask = (pred_clusters == cluster_index)
                samples_in_this_cluster = self.train_samples_matrix[mask]
                assert self.n_features == 2  # 为了方便和可行性，只实现了对二维特征的样本的可视化
                plt.scatter(samples_in_this_cluster[:, 0], samples_in_this_cluster[:, 1],
                            marker=marker_list[cluster_index],
                            color=color_list[cluster_index], s=20, label='cluster ' + str(cluster_index))
            plt.scatter(mean_vectors[:, 0], mean_vectors[:, 1],
                        marker="o", color="black", s=40, label="predict center")
            plt.scatter(self.init_mean_vectors[:, 0], self.init_mean_vectors[:, 1],
                        marker="x", color="grey", s=40, label="initial center")
            if self.clustered_samples_list is not None:
                actual_k = len(self.clustered_samples_list)
                generate_mean_vectors = np.empty((actual_k, self.n_features))
                for cluster_index in range(actual_k):
                    generate_mean_vectors[cluster_index] = np.mean(self.clustered_samples_list[cluster_index], axis=0)
                plt.scatter(generate_mean_vectors[:, 0], generate_mean_vectors[:, 1], marker="x", color="green", s=40,
                            label="generate center")
            title = "GMM Clustered Samples Graph(k=" + str(self.k) + ", init"
            if self.init_method == "random":
                title += " randomly)"
            elif self.init_method == "heuristic":
                title += " heuristically)"
            plt.title(title)
            plt.legend(loc="best")
            plt.xlabel("d_1")
            plt.ylabel("d_2")
            plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
            plt.show()

        return pred_clusters, mean_vectors, covariance_matrixes, alphas, actual_iter_times

    def cluster(self, draw_result=False):
        if self.verbose:
            print("clustering using gaussian mixture model:")
        return self.__cluster_by_gmm(draw_result)
