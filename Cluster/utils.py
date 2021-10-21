# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
is_show = True  # 控制是否绘图


# 生成数据
def generate_data(k, n_features=2, sample_num_list: list = None, mean_vector_list: list = None,
                  covariance_matrix_list: list = None):
    DEBUG = False

    assert k > 1 and n_features > 0

    if sample_num_list is not None:
        assert len(sample_num_list) == k
    else:
        sample_num_list = [100 for _ in range(k)]
    if DEBUG:
        print("sample_num_list:", sample_num_list)

    if mean_vector_list is not None:
        assert len(mean_vector_list) == k
        for mean_vector in mean_vector_list:
            assert mean_vector.shape == (n_features,)
    else:
        mean_vector_list = []
        for cluster_index in range(k):
            mean_vector_list.append(np.random.normal(cluster_index - k // 2, 0.1, n_features))
    if DEBUG:
        print("mean_vector_list:")
        for cluster_index in range(k):
            print("cluster ", str(cluster_index), '\n', mean_vector_list[cluster_index])

    if covariance_matrix_list is not None:
        assert len(covariance_matrix_list) == k
        for covariance_matrix in covariance_matrix_list:
            assert covariance_matrix.shape == (n_features, n_features)
            assert np.allclose(covariance_matrix, covariance_matrix.T)
    else:
        covariance_matrix_list = []
        for cluster_index in range(k):
            sigma_array = 0.25 * np.random.rand(n_features)
            covariance_matrix = np.diag(sigma_array)
            covariance_matrix_list.append(covariance_matrix)
    if DEBUG:
        print("covariance_matrix_list:")
        for cluster_index in range(k):
            print("cluster ", str(cluster_index), '\n', covariance_matrix_list[cluster_index])

    samples_list = []
    for cluster_index in range(k):
        samples_list.append(np.random.multivariate_normal(mean=mean_vector_list[cluster_index],
                                                          cov=covariance_matrix_list[cluster_index],
                                                          size=sample_num_list[cluster_index]))
    if DEBUG:
        print("generate data finished.")
        for i in range(1 + sample_num_list[1] // 20):
            print(samples_list[1][i, :])
        for i in range(1 + sample_num_list[-1] // 20):
            print(samples_list[-1][-i, :])

    assert len(mean_vector_list) == len(covariance_matrix_list) == len(samples_list) == k
    mixed_samples = np.vstack(samples_list)  # 混合所有样本，隐藏按顺序排列的簇信息
    np.random.shuffle(mixed_samples)  # 打乱
    assert mixed_samples.shape == (np.sum(np.array(sample_num_list)), n_features)
    return mixed_samples, samples_list


# 绘制生成数据情况的示意图
def draw_data_generate(samples_list):
    if is_show:
        k = len(samples_list)
        assert 0 < k <= 6  # 为了画图颜色和图例的方便，最多支持6个簇
        marker_list = [".", "*", "^", "p", "+", "D"]
        color_list = ["tomato", "deepskyblue", "orange", "violet", "green", "teal"]
        for cluster_index in range(k):
            samples = samples_list[cluster_index]
            samples_num, n_features = samples.shape
            assert n_features == 2  # 为了方便和可行性，只实现了对二维特征的样本的可视化
            plt.scatter(samples[:, 0], samples[:, 1], marker=marker_list[cluster_index],
                        color=color_list[cluster_index], s=20, label='cluster ' + str(cluster_index))
        title = "Sample Scatter Graph(k=" + str(k) + ")"
        plt.title(title)
        plt.legend(loc="best")
        plt.xlabel("d_1")
        plt.ylabel("d_2")
        # plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
        plt.show()


if __name__ == '__main__':
    # samples_list = generate_data(k=3, n_features=2)
    mixed_samples, samples_list = generate_data(k=3, n_features=2,
                                                mean_vector_list=[np.array([2, 4]), np.array([3, -6]),
                                                                  np.array([-2, - 4])],
                                                covariance_matrix_list=[np.array([[0.3, 0], [0, 0.6]]),
                                                                        np.array([[0.5, 0], [0, 0.5]]),
                                                                        np.array([[0.55, 0], [0, 0.35]])])
    draw_data_generate(samples_list)
