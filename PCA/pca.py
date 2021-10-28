# -*- coding: UTF-8 -*-
from utils import *


class PCA_Model:
    def __init__(self, k, origin_samples):
        self.k = k
        self.origin_samples = origin_samples
        self.origin_dim, self.n_samples = origin_samples.shape
        assert self.n_samples > 0 and self.origin_dim > 0 and 0 < k < self.origin_dim

    def _centralize(self, origin_samples):
        origin_mean_vector = np.mean(origin_samples, axis=1)
        centralized_samples = (origin_samples.T - origin_mean_vector).T
        assert origin_mean_vector.shape == (self.origin_dim,)
        assert centralized_samples.shape == (self.origin_dim, self.n_samples)
        return centralized_samples, origin_mean_vector

    def _reconstruct(self, selected_eigen_vectors, centralized_origin_samples, origin_mean_vector):
        reduced_samples = np.zeros((self.n_samples, self.origin_dim))
        # for i in range(self.n_samples):
        #     for j in range(self.k):
        #         w = selected_eigen_vectors[:, j]
        #         x = centralized_origin_samples[:, i]
        #         assert w.shape == x.shape == (self.origin_dim,)
        #         reduced_samples[i] += np.dot(w, x) * w
        reduced_samples = np.matmul(np.matmul(selected_eigen_vectors, selected_eigen_vectors.T),
                                    centralized_origin_samples)
        reduced_samples = (reduced_samples.T + origin_mean_vector).T
        assert reduced_samples.shape == (self.origin_dim, self.n_samples)
        return reduced_samples

    def adapt_pca(self, method_to_compute_eigen_vectors="SVD"):
        # 对样本矩阵进行中心化
        centralized_origin_samples, origin_mean_vector = self._centralize(self.origin_samples)
        assert centralized_origin_samples.shape == (self.origin_dim, self.n_samples)
        assert origin_mean_vector.shape == (self.origin_dim,)
        if method_to_compute_eigen_vectors == "SVD":
            # 对样本矩阵进行特征值分解(EVD)
            try:
                U, S, V_T = np.linalg.svd(centralized_origin_samples)
                assert U.shape == (self.origin_dim, self.origin_dim)
                assert V_T.shape == (self.n_samples, self.n_samples)
            except np.linalg.LinAlgError:
                print("奇异值分解错误！")
            else:
                # 取前k个最大特征值对应的特征向量
                selected_eigen_vectors = U[:, :self.k]
                assert selected_eigen_vectors.shape == (self.origin_dim, self.k)
                # 一旦降维维度超过某个值，特征向量矩阵将出现复向量，对其保留实部
                # selected_eigen_vectors = np.real(selected_eigen_vectors)  # TODO: 这里应该是不需要的
                reduced_samples = self._reconstruct(selected_eigen_vectors, centralized_origin_samples,
                                                    origin_mean_vector)
                return reduced_samples, selected_eigen_vectors, origin_mean_vector, centralized_origin_samples
        elif method_to_compute_eigen_vectors == "EVD":
            # 计算协方差矩阵
            covariance_matrix = np.matmul(centralized_origin_samples, centralized_origin_samples.T)
            assert covariance_matrix.shape == (self.origin_dim, self.origin_dim)
            # 对协方差矩阵进行特征值分解(EVD)
            try:
                eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
                assert eigen_values.shape == (self.origin_dim,)
                assert eigen_vectors.shape == (self.origin_dim, self.origin_dim)
            except np.linalg.LinAlgError:
                print("特征值分解错误！")
            else:
                # 对特征值排序，获得排序后的下标
                sorted_eigen_values_indexes = np.argsort(eigen_values)
                # 取前k个最大特征值对应的特征向量
                selected_indexes = sorted_eigen_values_indexes[:-(self.k + 1):-1]
                assert selected_indexes.shape == (self.k,)
                selected_eigen_vectors = eigen_vectors[:, selected_indexes]
                assert selected_eigen_vectors.shape == (self.origin_dim, self.k)
                reduced_samples = self._reconstruct(selected_eigen_vectors, centralized_origin_samples,
                                                    origin_mean_vector)
                return reduced_samples, selected_eigen_vectors, origin_mean_vector, centralized_origin_samples
        else:
            raise NotImplementedError  # not implemented
