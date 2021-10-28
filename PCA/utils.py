# -*- coding: UTF-8 -*-
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_lfw_people
from matplotlib import pyplot as plt
import math
from PCA import pca

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
is_show = True  # 控制是否绘图


# 生成数据
def generate_data(n_samples, n_features=3, mean_vector=None, covariance_matrix=None):
    DEBUG = False

    assert n_samples > 0 and n_features > 0

    if mean_vector is not None:
        assert len(mean_vector) == n_features
    else:
        mean_vector = 2 * np.random.normal(size=n_features)
    if DEBUG:
        print("mean_vector:", mean_vector)

    if covariance_matrix is not None:
        assert covariance_matrix.shape == (n_features, n_features)
        assert np.allclose(covariance_matrix, covariance_matrix.T)
    else:
        sigmas = 0.5 * np.ones(n_features)
        if n_features > 2:
            sigmas[2:] *= 0.01
        else:
            sigmas[1:] *= 0.01
        covariance_matrix = np.diag(sigmas)
    if DEBUG:
        print("covariance_matrix:")
        print(covariance_matrix)

    samples = np.random.multivariate_normal(mean=mean_vector, cov=covariance_matrix, size=n_samples).T
    assert samples.shape == (n_features, n_samples)

    if DEBUG:
        print("generate data finished.")
        # print(samples)

    return samples


# 绘制生成数据情况的示意图
def draw_data_generate(samples):
    if is_show:
        n_features, n_samples = samples.shape
        if n_features == 2:
            plt.scatter(samples[0, :], samples[1, :], marker=".", color="tomato", s=20, label='generate data')
            title = "Samples Scatter Graph{(n_samples,n_features)=(" + str(n_samples) + "," + str(n_features) + ")}"
            plt.title(title, fontsize=16)
            plt.xlabel("x")
            plt.ylabel("y")
        elif n_features == 3:
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            ax.scatter(samples[0, :], samples[1, :], samples[2, :], facecolor='tomato', label='generate data')
            title = "Samples Scatter Graph{(n_samples,n_features)=(" + str(n_samples) + "," + str(n_features) + ")}"
            ax.set_title(title, fontsize=16)
            plt.legend(loc="best")
            ax.set_zlabel('$z$', fontdict={'size': 14, 'color': 'black'})
            ax.set_ylabel('$y$', fontdict={'size': 14, 'color': 'black'})
            ax.set_xlabel('$x$', fontdict={'size': 14, 'color': 'black'})
        else:
            assert False  # 只支持2维数据或3维数据的可视化

        plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
        plt.show()


def draw_pca_result(origin_samples, reduced_samples, selected_eigen_vectors, origin_mean_vector,
                    centralized_origin_samples):
    if is_show:
        assert origin_samples.shape[0] == reduced_samples.shape[0]
        origin_dim, n_samples = origin_samples.shape
        k = reduced_samples.shape[1]
        if origin_dim == 2:
            x = [origin_mean_vector[0] - 4 * selected_eigen_vectors[0],
                 origin_mean_vector[0] + 4 * selected_eigen_vectors[0]]
            y = [origin_mean_vector[1] - 4 * selected_eigen_vectors[1],
                 origin_mean_vector[1] + 4 * selected_eigen_vectors[1]]
            plt.plot(x, y, color="deepskyblue", linewidth=1.0, linestyle='-', label="projection direction")
            plt.scatter(origin_samples[0, :], origin_samples[1, :], marker="x", color="tomato", s=20,
                        label='origin data')
            plt.scatter(reduced_samples[0, :], reduced_samples[1, :], marker=".", color="blue", s=20,
                        label='projected data')
            title = "PCA Result Graph((n_samples,origin_dim)=(" + str(n_samples) + "," + str(origin_dim) + "))"
            plt.title(title)
            plt.legend(loc="best")
            plt.xlabel("x")
            plt.ylabel("y")
        elif origin_dim == 3:
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            x = [origin_mean_vector[0] - 4 * selected_eigen_vectors[0, 0],
                 origin_mean_vector[0] + 4 * selected_eigen_vectors[0, 0]]
            y = [origin_mean_vector[1] - 4 * selected_eigen_vectors[1, 0],
                 origin_mean_vector[1] + 4 * selected_eigen_vectors[1, 0]]
            z = [origin_mean_vector[2] - 4 * selected_eigen_vectors[2, 0],
                 origin_mean_vector[2] + 4 * selected_eigen_vectors[2, 0]]
            ax.plot(x, y, z, color='deepskyblue', label='eigen_vector_direction_1', alpha=1)
            x = [origin_mean_vector[0] - 4 * selected_eigen_vectors[0, 1],
                 origin_mean_vector[0] + 4 * selected_eigen_vectors[0, 1]]
            y = [origin_mean_vector[1] - 4 * selected_eigen_vectors[1, 1],
                 origin_mean_vector[1] + 4 * selected_eigen_vectors[1, 1]]
            z = [origin_mean_vector[2] - 4 * selected_eigen_vectors[2, 1],
                 origin_mean_vector[2] + 4 * selected_eigen_vectors[2, 1]]
            ax.plot(x, y, z, color='olive', label='eigen_vector_direction_2', alpha=1)
            ax.scatter(origin_samples[0, :], origin_samples[1, :], origin_samples[2, :], facecolor='tomato',
                       label='origin data', marker="x")
            ax.scatter(reduced_samples[0, :], reduced_samples[1, :], reduced_samples[2, :], facecolor='blue',
                       label='projected data', marker="o")
            title = "PCA Result Graph((n_samples,origin_dim)=(" + str(n_samples) + "," + str(origin_dim) + "))"
            ax.set_title(title)
            plt.legend(loc="best")
            ax.set_zlabel('$z$', fontdict={'size': 14, 'color': 'black'})
            ax.set_ylabel('$y$', fontdict={'size': 14, 'color': 'black'})
            ax.set_xlabel('$x$', fontdict={'size': 14, 'color': 'black'})
        else:
            assert False  # 只支持2维数据或3维数据的可视化

        plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
        plt.show()


# 计算峰值信噪比
def psnr(source, target):
    assert source.shape == target.shape
    mse = np.mean((source - target) ** 2)
    # if mse < 1.0e-10:
    #     return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    # # 测试生成数据
    # samples = generate_data(n_samples=50, n_features=2, mean_vector=None, covariance_matrix=None)  # 三维数据
    # draw_data_generate(samples)
    # samples = generate_data(n_samples=100, n_features=3, mean_vector=None, covariance_matrix=None)  # 二维数据
    # draw_data_generate(samples)
    # os.system("pause")

    # 在生成数据集上测试PCA(origin_dim=2)
    mean_vector = np.array([-3, 4])
    covariance_matrix = np.array([[1, 0], [0, 0.01]])
    origin_samples = generate_data(n_samples=50, n_features=2, mean_vector=mean_vector,
                                   covariance_matrix=covariance_matrix)  # 二维数据
    pca_model = pca.PCA_Model(k=1, origin_samples=origin_samples)
    reduced_samples, selected_eigen_vectors, origin_mean_vector, centralized_origin_samples = pca_model.adapt_pca("EVD")
    print("origin_mean_vector:", "\n", origin_mean_vector)
    print("selected_eigen_vectors:", "\n", selected_eigen_vectors)
    print("reduced_samples.shape:", reduced_samples.shape)
    draw_pca_result(origin_samples, reduced_samples, selected_eigen_vectors, origin_mean_vector,
                    centralized_origin_samples)

    reduced_samples, selected_eigen_vectors, origin_mean_vector, centralized_origin_samples = pca_model.adapt_pca("SVD")
    print("origin_mean_vector:", "\n", origin_mean_vector)
    print("selected_eigen_vectors:", "\n", selected_eigen_vectors)
    print("reduced_samples.shape:", reduced_samples.shape)
    draw_pca_result(origin_samples, reduced_samples, selected_eigen_vectors, origin_mean_vector,
                    centralized_origin_samples)
    os.system("pause")

    # 在生成数据集上测试PCA(origin_dim=3)
    mean_vector = np.array([2, -4, 7])
    covariance_matrix = np.array([[1, 0.01, 0.03], [0.01, 0.01, 0.02], [0.03, 0.02, 1]])
    origin_samples = generate_data(n_samples=50, n_features=3, mean_vector=mean_vector,
                                   covariance_matrix=covariance_matrix)  # 三维数据
    pca_model = pca.PCA_Model(k=2, origin_samples=origin_samples)
    reduced_samples, selected_eigen_vectors, origin_mean_vector, centralized_origin_samples = pca_model.adapt_pca("EVD")
    print("origin_mean_vector:", "\n", origin_mean_vector)
    print("selected_eigen_vectors:", "\n", selected_eigen_vectors)
    print("reduced_samples.shape:", reduced_samples.shape)
    draw_pca_result(origin_samples, reduced_samples, selected_eigen_vectors, origin_mean_vector,
                    centralized_origin_samples)

    reduced_samples, selected_eigen_vectors, origin_mean_vector, centralized_origin_samples = pca_model.adapt_pca("SVD")
    print("origin_mean_vector:", "\n", origin_mean_vector)
    print("selected_eigen_vectors:", "\n", selected_eigen_vectors)
    print("reduced_samples.shape:", reduced_samples.shape)
    draw_pca_result(origin_samples, reduced_samples, selected_eigen_vectors, origin_mean_vector,
                    centralized_origin_samples)

    # 图像处理任务上的实验
    faces = fetch_lfw_people(min_faces_per_person=60, color=False)
    origin_data = faces.data[:100, :] / 255
    print("origin data shape:", origin_data.shape)
    fig, axes = plt.subplots(3, 3, figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces.images[i, :, :], cmap="gray")
    plt.show()
    target_dim_list = range(1, 150, 10)
    psnr_list = []
    for target_dim in target_dim_list:
        pca_model = pca.PCA_Model(target_dim, origin_data.T)
        reduced_data_svd, selected_eigen_vectors_svd, origin_mean_vector_svd, centralized_origin_data_svd = pca_model.adapt_pca(
            "SVD")
        reduced_data_svd = reduced_data_svd.T
        psnr_list.append(psnr(reduced_data_svd, origin_data))
        if target_dim == 1 or target_dim == 51 or target_dim == 101:
            fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': [], 'yticks': []})
            for i in range(5):
                axes[0, i].imshow(origin_data[i, :].reshape(62, 47), cmap='gray')
                axes[1, i].imshow(reduced_data_svd[i, :].reshape(62, 47), cmap='gray')
            fname = "image comparison before and after pca(target_dim = " + str(target_dim) + ")"
            plt.savefig(fname=fname + ".svg", dpi=2000, format="svg")
            plt.show()
    plt.plot(target_dim_list, psnr_list, color="deepskyblue", linewidth=1.0, linestyle='-', label="psnr")
    plt.scatter(target_dim_list, psnr_list, marker=".", color="blue", s=20, label='psnr data')
    plt.xlabel("target dim")
    plt.ylabel("psnr")
    plt.legend(loc="best")
    title = "psnr-target dim graph"
    plt.title(title)
    plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
    plt.show()
