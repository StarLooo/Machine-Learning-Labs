# -*- coding: UTF-8 -*-
import os
import abc
from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
from Logistic_Regression import logistic_regression as LG

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
is_show = True  # 控制是否绘图


def generate_data(n_samples, n_features=2, pos_rate: float = 0.5, satisfy_naive_bayesian_hypothesis: bool = True,
                  pos_mean_vector=None, neg_mean_vector=None, covariance_matrix=None):
    DEBUG = False

    assert n_samples > 0 and n_features > 0 and 1 > pos_rate > 0
    if pos_mean_vector is not None:
        assert len(pos_mean_vector) == n_features
    else:
        pos_mean_vector = np.random.randn(n_features)
    if DEBUG:
        print("pos mean_vector:", pos_mean_vector)

    if neg_mean_vector is not None:
        assert len(pos_mean_vector) == n_features
    else:
        neg_mean_vector = -pos_mean_vector
    if DEBUG:
        print("neg mean_vector:", neg_mean_vector)

    if covariance_matrix is not None:
        assert covariance_matrix.shape == (n_features, n_features)
        assert np.allclose(covariance_matrix, covariance_matrix.T)
        if satisfy_naive_bayesian_hypothesis:
            assert np.allclose(covariance_matrix, np.diag(np.diagonal(covariance_matrix))) is True
    else:
        sigma_array = np.random.rand(n_features)
        if satisfy_naive_bayesian_hypothesis:
            covariance_matrix = np.diag(sigma_array)
        else:
            covariance_matrix = np.fill_diagonal(np.triu(np.random.rand(n_features, n_features)), sigma_array)
    if DEBUG:
        print("covariance_matrix:", covariance_matrix)

    pos_samples_num = np.ceil(n_samples * pos_rate).astype(np.int32)
    neg_samples_num = n_samples - pos_samples_num
    X = np.zeros((n_samples, n_features))  # 样本矩阵
    y = np.zeros(n_samples, dtype=np.int32)  # 标签向量
    # 生成正例
    X[:pos_samples_num, :] = np.random.multivariate_normal(pos_mean_vector, covariance_matrix,
                                                           size=pos_samples_num)
    y[:pos_samples_num] = 1
    # 生成负例
    X[pos_samples_num:, :] = np.random.multivariate_normal(neg_mean_vector, covariance_matrix,
                                                           size=neg_samples_num)
    y[pos_samples_num:] = 0

    if DEBUG:
        print("generate data finished.")
        for i in range(1 + n_samples // 100):
            print(X[i, :], y[i])
        for i in range(1 + n_samples // 100):
            print(X[-i, :], y[-i])

    return X, y, pos_samples_num, neg_samples_num


# 绘制生成数据情况的示意图
def draw_data_generate(X, pos_samples_num, neg_samples_num):
    if is_show:
        # 训练集中样本散点图
        plt.scatter(X[:pos_samples_num, 0], X[:pos_samples_num, 1], marker='.', color='red', s=20, label='pos sample')
        plt.scatter(X[pos_samples_num:, 0], X[pos_samples_num:, 1], marker='.', color='blue', s=20, label='neg sample')
        plt.title("pos samples num=" + str(pos_samples_num) + ",neg samples num=" + str(neg_samples_num))
        plt.legend(loc="best")
        plt.show()


# 训练结束后对训练结果进行可视化
def draw_predict_analysis(X_train, train_pos_samples_num, X_test, test_pos_samples_num, w, title):
    left = min(np.min(X_train[:, 0]), np.min(X_test[:, 0]))
    right = max(np.max(X_train[:, 0]), np.max(X_test[:, 0]))
    draw_x = np.linspace(left, right, 200)
    draw_y = np.array([-(w[2] + w[0] * x) / w[1] for x in draw_x])

    # 训练集中样本散点图
    plt.scatter(X_train[:train_pos_samples_num, 0], X_train[:train_pos_samples_num, 1], marker='.', color='red', s=20,
                label='pos train sample')
    plt.scatter(X_train[train_pos_samples_num:, 0], X_train[train_pos_samples_num:, 1], marker='.', color='blue', s=20,
                label='neg train sample')
    # 训练得到的划分直线
    plt.plot(draw_x, draw_y, color='black', linewidth=1.0, linestyle='--',
             label="divide line")
    plt.title(title + "(pos train samples_num=" + str(train_pos_samples_num) + ",neg train samples_num=" + str(
        len(X_train) - train_pos_samples_num) + ")")
    plt.legend(loc='best')
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.show()

    # 测试集中样本散点图
    plt.scatter(X_test[:test_pos_samples_num, 0], X_test[:test_pos_samples_num, 1], marker='o', color='red', s=10,
                label='pos test sample')
    plt.scatter(X_test[test_pos_samples_num:, 0], X_test[test_pos_samples_num:, 1], marker='o', color='blue', s=10,
                label='neg test sample')
    # 训练得到的划分直线
    plt.plot(draw_x, draw_y, color='black', linewidth=1.0, linestyle='--',
             label="divide line")
    plt.title(title + "(pos test samples_num=" + str(test_pos_samples_num) + ",neg test samples_num=" + str(
        len(X_test) - test_pos_samples_num) + ")")
    plt.legend(loc='best')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, train_param, hyper_params, verbose=False):
        self.train_param = train_param
        self.hyper_params = hyper_params
        self.verbose = verbose

    @abc.abstractmethod
    def train(self):
        pass


class Gradient_Descent_Optimizer(Optimizer, ABC):
    def __init__(self, train_param, hyper_params, loss_func, grad_func, verbose=False):
        super().__init__(train_param, hyper_params, verbose)
        self.lr, self.max_iter_times, self.epsilon = hyper_params
        assert self.lr > 0. and self.max_iter_times > 0 and self.epsilon >= 0.
        self.loss_func = loss_func
        self.grad_func = grad_func

    def train(self):
        if self.verbose:
            print("optimize with gradient descent.")
        train_loss = self.loss_func(self.train_param)
        train_loss_list = []
        latest_grad = None
        actual_iter_times = 0
        for iter_num in range(1, self.max_iter_times + 1):
            actual_iter_times = iter_num
            pre_loss = train_loss  # 上一次迭代的loss
            train_loss_list.append(train_loss)  # 记录train_loss
            pre_param = self.train_param  # 上一次迭代的w
            latest_grad = self.grad_func(self.train_param)  # 求梯度
            # 若梯度在误差允许范围内接近0则结束训练，退出循环
            if np.linalg.norm(latest_grad, 2) < self.epsilon:
                if self.verbose:
                    print("gradient descent finished, actual iter times:", actual_iter_times)
                break
            new_param = self.train_param - self.lr * latest_grad  # 梯度下降
            train_loss = self.loss_func(new_param)  # 计算本次迭代后的训练误差
            # 若loss不再下降，则不更新参数，并减小学习率
            if train_loss >= pre_loss:
                self.lr *= 0.5  # 减小学习率
                train_loss = pre_loss
                pass
            else:
                # 否则更新参数
                self.train_param = new_param
            # 若迭代次数达到上限，训练结束
            if actual_iter_times == self.max_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list, latest_grad


# 寻找最优学习率
def find_best_lr(train_method="gradient descent"):
    # 第一次搜索
    lr_range = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    # 第二次搜索
    # lr_range = [0.65, 0.75, 0.85, 0.95]
    max_iters = 2000
    n_train = 200
    n_test = 1000
    pos_mean_vector = np.array([1.4, 2.1])
    neg_mean_vector = np.array([-0.8, -1.5])
    hypothesis = True
    covariance_matrix = np.array([[0.4, 0.0], [0.0, 0.5]])
    X_train, y_train, train_pos_samples_num, train_neg_samples_num = generate_data(n_samples=n_train,
                                                                                   pos_mean_vector=pos_mean_vector,
                                                                                   neg_mean_vector=neg_mean_vector,
                                                                                   satisfy_naive_bayesian_hypothesis=hypothesis,
                                                                                   covariance_matrix=covariance_matrix)
    X_test, y_test, test_pos_samples_num, test_neg_samples_num = generate_data(n_samples=n_test,
                                                                               pos_mean_vector=pos_mean_vector,
                                                                               neg_mean_vector=neg_mean_vector,
                                                                               satisfy_naive_bayesian_hypothesis=hypothesis,
                                                                               covariance_matrix=covariance_matrix)
    draw_color_list = ["red", "blue", "green", "black", "yellow", "purple", "pink"]
    for i in range(len(lr_range)):
        lr = lr_range[i]
        draw_color = draw_color_list[i]
        epsilon = 1e-3
        logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                                test_pos_samples_num=test_pos_samples_num,
                                                train_neg_samples_num=train_neg_samples_num,
                                                test_neg_samples_num=test_neg_samples_num,
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                regular_coef=0.0, verbose=True)
        train_loss_list = []
        iter_times = 0
        train_param = [lr, max_iters, epsilon]
        if train_method == "gradient descent":
            w, train_rmse, test_rmse, iter_times, train_loss_list = logistic.train(train_method, train_param,
                                                                                   draw_result=False)
        else:
            raise NotImplementedError
        print("train loss:", train_loss_list[-1])
        assert len(train_loss_list) == iter_times + 1
        plt.plot(range(20, iter_times + 1), np.array(train_loss_list)[20:], color=draw_color, linewidth=1.0,
                 linestyle='-', label="lr=" + str(lr))
    plt.xlabel("iter_times")
    plt.ylabel("train_loss")
    plt.legend(loc="best")
    plt.title("loss-iter graph(" + train_method + ")")
    plt.savefig(fname="compare_lr(" + train_method + ").svg", dpi=10000, format="svg")
    plt.show()


if __name__ == '__main__':
    # test find_best_lr()
    # find_best_lr()

    # os.system("pause")

    # Test generate_data() and draw_data_generate()
    # generate_data(n_samples=1000, n_features=10)
    n_train = 200
    n_test = 1000
    # pos_mean_vector = np.array([1., 1.2])
    # neg_mean_vector = np.array([-1., -1.2])
    pos_mean_vector = np.array([1.4, 2.1])
    neg_mean_vector = np.array([-0.8, -1.5])
    hypothesis = True
    covariance_matrix = np.array([[0.3, 0.0], [0.0, 0.4]])
    X_train, y_train, train_pos_samples_num, train_neg_samples_num = generate_data(n_samples=n_train,
                                                                                   pos_mean_vector=pos_mean_vector,
                                                                                   neg_mean_vector=neg_mean_vector,
                                                                                   satisfy_naive_bayesian_hypothesis=hypothesis,
                                                                                   covariance_matrix=covariance_matrix)
    X_test, y_test, test_pos_samples_num, test_neg_samples_num = generate_data(n_samples=n_test,
                                                                               pos_mean_vector=pos_mean_vector,
                                                                               neg_mean_vector=neg_mean_vector,
                                                                               satisfy_naive_bayesian_hypothesis=hypothesis,
                                                                               covariance_matrix=covariance_matrix)
    # draw_data_generate(X_train, train_pos_samples_num, train_neg_samples_num)
    # draw_data_generate(X_test, test_pos_samples_num, test_neg_samples_num)

    logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                            test_pos_samples_num=test_pos_samples_num,
                                            train_neg_samples_num=train_neg_samples_num,
                                            test_neg_samples_num=test_neg_samples_num,
                                            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                            regular_coef=0.0, verbose=True)
    w, train_loss, test_loss, actual_iter_times, train_loss_list = logistic.train(train_method="gradient descent",
                                                                                  train_param=[0.5, 10000, 1e-3],
                                                                                  draw_result=True)
