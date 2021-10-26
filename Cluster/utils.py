# -*- coding: UTF-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from Cluster import cluster_model

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
            sigma_array = 0.1 * np.random.rand(n_features)
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
    mixed_samples = np.vstack(samples_list)  # 混合所有样本
    ref_clusters = []
    for cluster_index in range(k):
        ref_clusters.append([cluster_index for _ in range(sample_num_list[cluster_index])])
    indexes = np.arrange(0, len(sample_num_list))
    np.random.shuffle(indexes)  # 打乱
    ref_clusters = np.array(ref_clusters[indexes])
    mixed_samples = mixed_samples[indexes]

    assert mixed_samples.shape == (np.sum(np.array(sample_num_list)), n_features)
    assert ref_clusters.shape == (np.sum(np.array(sample_num_list)),)
    return mixed_samples, ref_clusters


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
                        color=color_list[cluster_index], s=20, label='generate cluster ' + str(cluster_index))
        title = "Samples Scatter Graph(k=" + str(k) + ")"
        plt.title(title)
        plt.legend(loc="best")
        plt.xlabel("d_1")
        plt.ylabel("d_2")
        # plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
        plt.show()


# 研究不同超参k对轮廓系数的影响
def show_different_k(method):
    if is_show:
        select_k_range = range(2, 10)
        actual_k_range = range(3, 6)
        n_features = 2
        params = [100, 1e-12]
        for actual_k in actual_k_range:
            train_samples_matrix, clustered_samples_list = generate_data(actual_k, n_features)
            # draw_data_generate(clustered_samples_list)
            verbose = False
            sc_list = []
            for select_k in select_k_range:
                if method == "k-means":
                    k_means = cluster_model.K_Means(select_k, n_features, params, train_samples_matrix,
                                                    clustered_samples_list,
                                                    "heuristic", verbose)
                    pred_clusters, _, _, _ = k_means.cluster(draw_result=False)
                    sc = k_means.compute_silhouette_coefficient(pred_clusters)
                    sc_list.append(sc)
                elif method == "gmm":
                    gmm = cluster_model.Gaussian_Mixture_Model(select_k, n_features, params, train_samples_matrix,
                                                               clustered_samples_list, "heuristic", verbose)
                    pred_clusters, _, _, _, _ = gmm.cluster(draw_result=False)
                    sc = gmm.compute_silhouette_coefficient(pred_clusters)
                    sc_list.append(sc)
                else:
                    raise NotImplementedError
            plt.scatter(select_k_range, sc_list, marker='o', color='blue', s=10)
            plt.plot(select_k_range, sc_list, color='blue', linewidth=1.0, linestyle='-')
            plt.xlabel("select k")
            plt.ylabel("SC")
            title = "SC of " + method + " in different select k(actual k =" + str(actual_k) + ")"
            plt.title(title)
            # plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
            plt.show()


# 研究不同初始化条件下，GMM和K-Means的聚类效果对比
def compare_diff_cluster_method(init_method, repetitions=20):
    if is_show:
        k = 4
        n_features = 2
        params = [5000, 1e-12]
        mean_vector_list = [np.array([-0.5, 0]), np.array([1, 2]), np.array([3.5, 1.5]), np.array([1, 4.5])]
        covariance_matrix_list = [np.array([[0.2, 0], [0, 0.25]]), np.array([[0.1, 0], [0, 0.25]]),
                                  np.array([[0.15, 0], [0, 0.25]]), np.array([[0.15, 0], [0, 0.05]])]
        sample_num_list = [150, 200, 100, 250]
        verbose = False
        gmm_sc_list = []
        k_means_sc_list = []
        for repeat_num in range(1, repetitions + 1):
            train_samples_matrix, clustered_samples_list = generate_data(4, n_features, sample_num_list,
                                                                         mean_vector_list,
                                                                         covariance_matrix_list)
            # draw_data_generate(clustered_samples_list)
            gmm = cluster_model.Gaussian_Mixture_Model(k, n_features, params, train_samples_matrix,
                                                       clustered_samples_list,
                                                       init_method, verbose)
            pred_clusters, mean_vectors, covariance_matrixes, alphas, actual_iter_times = gmm.cluster(draw_result=False)
            gmm_sc = gmm.compute_silhouette_coefficient(pred_clusters)
            gmm_sc_list.append(gmm_sc)
            k_means = cluster_model.K_Means(k, n_features, params, train_samples_matrix, clustered_samples_list,
                                            init_method, verbose)
            pred_clusters, mean_square_loss, mean_vectors, actual_iter_times = k_means.cluster(draw_result=False)
            k_means_sc = gmm.compute_silhouette_coefficient(pred_clusters)
            k_means_sc_list.append(k_means_sc)

        plt.scatter(range(1, repetitions + 1), gmm_sc_list, marker='o', color='blue', s=10, label="gmm sc")
        plt.plot(range(1, repetitions + 1), gmm_sc_list, color='blue', linewidth=1.0, linestyle='-')
        plt.scatter(range(1, repetitions + 1), k_means_sc_list, marker='o', color='red', s=10, label="k-means sc")
        plt.plot(range(1, repetitions + 1), k_means_sc_list, color='red', linewidth=1.0, linestyle='-')
        plt.legend(loc="best")
        plt.xlabel("repeat num")
        plt.ylabel("SC")
        plt.xlim(1, repetitions)
        title = "SC Of Different Cluster Method(k=4, repetitions=" + str(repetitions) + ", init"
        if init_method == "random":
            title += " randomly)"
        elif init_method == "heuristic":
            title += " heuristically)"
        plt.title(title)
        # plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
        plt.show()


if __name__ == '__main__':
    # 研究生成数据
    # mixed_samples, samples_list = generate_data(k=3, n_features=2,
    #                                             mean_vector_list=[np.array([2, 4]), np.array([3, -6]),
    #                                                               np.array([-2, - 4])],
    #                                             covariance_matrix_list=[np.array([[0.3, 0], [0, 0.6]]),
    #                                                                     np.array([[0.5, 0], [0, 0.5]]),
    #                                                                     np.array([[0.55, 0], [0, 0.35]])])
    # draw_data_generate(samples_list)
    # os.system("pause")

    # 研究初始化方式对K-Means聚类效果的影响
    # k = 3
    # n_features = 2
    # params = [100, 1e-12]
    # mean_vector_list = [np.array([2, 4]), np.array([3, -6]), np.array([-2, - 4])]
    # covariance_matrix_list = [np.array([[0.3, 0], [0, 0.6]]), np.array([[0.5, 0], [0, 0.5]]),
    #                           np.array([[0.55, 0], [0, 0.35]])]
    # sample_num_list = [100, 100, 100]
    # train_samples_matrix, clustered_samples_list = generate_data(3, n_features, sample_num_list, mean_vector_list,
    #                                                              covariance_matrix_list)
    # verbose = True
    #
    # k_means = cluster_model.K_Means(k, n_features, params, train_samples_matrix, clustered_samples_list, "random",
    #                                 verbose)
    # pred_clusters, _, _, _ = k_means.cluster(draw_result=True)
    # k_means = cluster_model.K_Means(k, n_features, params, train_samples_matrix, clustered_samples_list, "heuristic",
    #                                 verbose)
    # pred_clusters, _, _, _ = k_means.cluster(draw_result=True)
    # os.system("pause")

    # 研究超参k的选择
    # show_different_k(method="k-means")
    # show_different_k(method="gmm")
    # os.system("pause")

    # 研究GMM的效果
    # k = 3
    # n_features = 2
    # params = [1000, 1e-12]
    # mean_vector_list = [np.array([0.5, 0.5]), np.array([5.5, 2.5]), np.array([1, 7])]
    # covariance_matrix_list = [np.array([[1, 0], [0, 3]]), np.array([[2, 0], [0, 2]]),
    #                           np.array([[6, 0], [0, 2]])]
    # sample_num_list = [200, 300, 500]
    # train_samples_matrix, clustered_samples_list = generate_data(3, n_features, sample_num_list, mean_vector_list,
    #                                                              covariance_matrix_list)
    # verbose = True
    #
    # gmm = cluster_model.Gaussian_Mixture_Model(k, n_features, params, train_samples_matrix, clustered_samples_list,
    #                                            "random", verbose)
    # pred_clusters, mean_vectors, covariance_matrixes, alphas, actual_iter_times = gmm.cluster(draw_result=True)
    # os.system("pause")

    # 研究GMM和K-Means的聚类效果对比
    compare_diff_cluster_method(init_method="heuristic")
    os.system("pause")

    # 在wine真实数据集上进行实验
    # wine_data = load_wine()
    # select_k_range = range(2, 8)
    # actual_k = 3
    # # print(wine_data.DESCR)
    # train_samples_matrix = wine_data.data
    # train_samples_matrix = train_samples_matrix / train_samples_matrix.max(axis=0)  # 按列归一化
    # # print("train_samples_matrix:")
    # # print(train_samples_matrix)
    # # correlation_matrix  = pd.DataFrame(train_samples_matrix).corr()
    # # print(correlation_matrix)
    # pca = PCA(n_components=2)
    # decomposed_train_samples_matrix = pca.fit_transform(train_samples_matrix)
    # print(pca.explained_variance_ratio_)
    # print(decomposed_train_samples_matrix.shape)
    # n_samples, n_features = decomposed_train_samples_matrix.shape
    # print("train_samples_matrix.shape:", decomposed_train_samples_matrix.shape)
    # ref_labels = wine_data.target
    # # print("ref_labels")  # ref_labels表明原数据矩阵是已经按类排好的
    # # print(ref_labels)
    # clustered_samples_list = [decomposed_train_samples_matrix[ref_labels == i] for i in range(actual_k)]
    # # draw_data_generate(clustered_samples_list)
    # params = [100, 1e-12]
    # verbose = False
    # k_means_sc_list = []
    # gmm_sc_list = []
    # # 利用肘方法选择k
    # for select_k in select_k_range:
    #     k_means = cluster_model.K_Means(select_k, n_features, params, decomposed_train_samples_matrix,
    #                                     clustered_samples_list, "heuristic", verbose)
    #     pred_clusters, _, _, _ = k_means.cluster(draw_result=False)
    #     sc = k_means.compute_silhouette_coefficient(pred_clusters)
    #     k_means_sc_list.append(sc)
    #
    #     gmm = cluster_model.Gaussian_Mixture_Model(select_k, n_features, params, decomposed_train_samples_matrix,
    #                                                clustered_samples_list, "heuristic", verbose)
    #     pred_clusters, _, _, _, _ = gmm.cluster(draw_result=False)
    #     sc = gmm.compute_silhouette_coefficient(pred_clusters)
    #     gmm_sc_list.append(sc)
    # plt.scatter(select_k_range, k_means_sc_list, marker='o', color='blue', s=10, label="k-means sc")
    # plt.plot(select_k_range, k_means_sc_list, color='blue', linewidth=1.0, linestyle='-')
    # plt.scatter(select_k_range, gmm_sc_list, marker='o', color='red', s=10, label="gmm sc")
    # plt.plot(select_k_range, gmm_sc_list, color='red', linewidth=1.0, linestyle='-')
    # plt.legend(loc="best")
    # plt.xlabel("select k")
    # plt.ylabel("SC")
    # title = "SC of k-means and gmm in different select k in wine dataset"
    # plt.title(title)
    # # plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
    # plt.show()

    # 在iris真实数据集上进行实验
    # iris_data = load_iris()
    # select_k_range = range(2, 8)
    # actual_k = 3
    # print(iris_data.DESCR)
    # train_samples_matrix = iris_data.data
    # train_samples_matrix = train_samples_matrix / train_samples_matrix.max(axis=0)  # 按列归一化
    # # print("train_samples_matrix:")
    # # print(train_samples_matrix)
    # correlation_matrix  = pd.DataFrame(train_samples_matrix).corr()
    # print(correlation_matrix)
    # n_samples, n_features = train_samples_matrix.shape
    # print("train_samples_matrix.shape:", train_samples_matrix.shape)
    # ref_labels = iris_data.target
    # # print("ref_labels")  # ref_labels表明原数据矩阵是已经按类排好的
    # # print(ref_labels)
    # clustered_samples_list = [train_samples_matrix[ref_labels == i] for i in range(actual_k)]
    # # draw_data_generate(clustered_samples_list)
    # params = [100, 1e-12]
    # verbose = False
    # k_means_sc_list = []
    # gmm_sc_list = []
    # # 利用肘方法选择k
    # for select_k in select_k_range:
    #     k_means = cluster_model.K_Means(select_k, n_features, params, train_samples_matrix,
    #                                     clustered_samples_list, "heuristic", verbose)
    #     pred_clusters, _, _, _ = k_means.cluster(draw_result=False)
    #     sc = k_means.compute_silhouette_coefficient(pred_clusters)
    #     k_means_sc_list.append(sc)
    #
    #     gmm = cluster_model.Gaussian_Mixture_Model(select_k, n_features, params, train_samples_matrix,
    #                                                clustered_samples_list, "heuristic", verbose)
    #     pred_clusters, _, _, _, _ = gmm.cluster(draw_result=False)
    #     sc = gmm.compute_silhouette_coefficient(pred_clusters)
    #     gmm_sc_list.append(sc)
    # plt.scatter(select_k_range, k_means_sc_list, marker='o', color='blue', s=10, label="k-means sc")
    # plt.plot(select_k_range, k_means_sc_list, color='blue', linewidth=1.0, linestyle='-')
    # plt.scatter(select_k_range, gmm_sc_list, marker='o', color='red', s=10, label="gmm sc")
    # plt.plot(select_k_range, gmm_sc_list, color='red', linewidth=1.0, linestyle='-')
    # plt.legend(loc="best")
    # plt.xlabel("select k")
    # plt.ylabel("SC")
    # title = "SC of k-means and gmm in different select k in wine dataset"
    # plt.title(title)
    # # plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
    # plt.show()
    # os.system("pause")
