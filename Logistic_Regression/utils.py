# -*- coding: UTF-8 -*-
import abc
import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import datasets
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

from Logistic_Regression import logistic_regression as LG

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
is_show = True  # 控制是否绘图


# 生成数据
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
        plt.xlabel("d_1")
        plt.ylabel("d_2")
        plt.savefig(fname="draw_data_generate.svg", dpi=10000, format="svg")
        plt.show()


# 训练结束后对训练结果进行可视化
def draw_predict_analysis(X_train, train_pos_samples_num, X_test, test_pos_samples_num, w, title):
    left = min(np.min(X_train[:, 0]), np.min(X_test[:, 0]))
    right = max(np.max(X_train[:, 0]), np.max(X_test[:, 0]))
    draw_x = np.linspace((3 * left + right) / 4, (3 * right + left) / 4, 200)
    draw_y = np.array([-(w[0] + w[1] * x) / w[2] for x in draw_x])

    # 训练集中样本散点图
    plt.scatter(X_train[:train_pos_samples_num, 0], X_train[:train_pos_samples_num, 1], marker='.', color='red', s=20,
                label='pos train sample')
    plt.scatter(X_train[train_pos_samples_num:, 0], X_train[train_pos_samples_num:, 1], marker='.', color='blue', s=20,
                label='neg train sample')
    # 训练得到的划分直线
    plt.plot(draw_x, draw_y, color='black', linewidth=1.0, linestyle='--',
             label="divide line")
    title_1 = title + "(pos train samples_num=" + str(train_pos_samples_num) + ",neg train samples_num=" + str(
        len(X_train) - train_pos_samples_num) + ")"
    plt.title(title_1)
    plt.legend(loc='best')
    plt.xlabel("d_1")
    plt.ylabel("d_2")
    plt.savefig(fname=title_1 + ".svg", dpi=2000, format="svg")
    plt.show()

    # 测试集中样本散点图
    plt.scatter(X_test[:test_pos_samples_num, 0], X_test[:test_pos_samples_num, 1], marker='o', color='red', s=10,
                label='pos test sample')
    plt.scatter(X_test[test_pos_samples_num:, 0], X_test[test_pos_samples_num:, 1], marker='o', color='blue', s=10,
                label='neg test sample')
    # 训练得到的划分直线
    plt.plot(draw_x, draw_y, color='black', linewidth=1.0, linestyle='--',
             label="divide line")
    title_2 = title + "(pos test samples_num=" + str(test_pos_samples_num) + ",neg test samples_num=" + str(
        len(X_test) - test_pos_samples_num) + ")"
    plt.title(title_2)
    plt.legend(loc='best')
    plt.xlabel("d_1")
    plt.ylabel("d_2")
    plt.savefig(fname=title_2 + ".svg", dpi=2000, format="svg")
    plt.show()


# 优化器抽象接口
class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, train_param, hyper_params, verbose=False):
        self.train_param = train_param
        self.hyper_params = hyper_params
        self.verbose = verbose

    @abc.abstractmethod
    def train(self):
        pass


# 朴素梯度下降优化器
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
                    print("last lr:", self.lr)
                break
            new_param = self.train_param - self.lr * latest_grad  # 梯度下降
            train_loss = self.loss_func(new_param)  # 计算本次迭代后的训练误差
            # 若loss不再下降，则不更新参数，并减小学习率
            if train_loss >= pre_loss:
                self.lr *= 0.8  # 减小学习率
                train_loss = pre_loss
            else:
                # 否则更新参数
                self.train_param = new_param
                if self.lr < 2.0:
                    # 可以尝试增大学习率，加速收敛
                    self.lr *= 1.2
            # 若迭代次数达到上限，训练结束
            if actual_iter_times == self.max_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list, latest_grad


# 牛顿迭代优化器
class Newton_Optimizer(Optimizer, ABC):
    def __init__(self, train_param, hyper_params, loss_func, first_grad_func, second_grad_func, verbose=False):
        super().__init__(train_param, hyper_params, verbose)
        self.max_iter_times, self.epsilon = hyper_params
        assert self.max_iter_times > 0 and self.epsilon >= 0.
        self.loss_func = loss_func
        self.first_grad_func = first_grad_func
        self.second_grad_func = second_grad_func

    def train(self):
        if self.verbose:
            print("optimize with newton method.")
        train_loss = self.loss_func(self.train_param)
        train_loss_list = []
        first_grad = None
        second_grad = None
        actual_iter_times = 0
        for iter_num in range(1, self.max_iter_times + 1):
            actual_iter_times = iter_num
            pre_loss = train_loss  # 上一次迭代的loss
            train_loss_list.append(train_loss)  # 记录train_loss
            pre_param = self.train_param  # 上一次迭代的w
            first_grad = self.first_grad_func(self.train_param)  # 一阶导
            second_grad = self.second_grad_func(self.train_param)  # 二阶导
            # 若梯度在误差允许范围内接近0则结束训练，退出循环
            if np.linalg.norm(first_grad, 2) < self.epsilon:
                if self.verbose:
                    print("gradient descent finished, actual iter times:", actual_iter_times)
                break
            new_param = self.train_param - np.matmul(np.linalg.inv(second_grad), first_grad)  # 牛顿迭代
            train_loss = self.loss_func(new_param)  # 计算本次迭代后的训练误差
            self.train_param = new_param
            # 若迭代次数达到上限，训练结束
            if actual_iter_times == self.max_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list, first_grad, second_grad


# 画图分析，寻找最优学习率
def find_best_lr(train_method="gradient descent"):
    # 第一次搜索
    lr_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    # 第二次搜索
    # lr_range = [0.90, 0.95, 1.0, 1.05, 1.1, 1.15]
    max_iters = 20000
    n_train = 200
    n_test = 2000
    pos_mean_vector = np.array([1.0, 1.6])
    neg_mean_vector = np.array([-0.5, 0.0])
    hypothesis = True
    covariance_matrix = np.array([[0.4, 0.0], [0.0, 0.6]])
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
        epsilon = 1e-6
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
        plt.plot(range(5, iter_times + 1), np.array(train_loss_list)[5:], color=draw_color, linewidth=1.0,
                 linestyle='-', label="lr=" + str(lr))
    plt.xlabel("iter_times")
    plt.ylabel("train_loss")
    plt.legend(loc="best")
    plt.title("loss-iter graph(" + train_method + ")")
    plt.savefig(fname="compare_lr(" + train_method + ").svg", dpi=10000, format="svg")
    print("************************")
    plt.show()


# 可视化分析训练得到的分类结果(ROC曲线和混淆矩阵图)
def analysis(y_true, y_pred, mode):
    if is_show:
        # ROC曲线计算
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        # AUC值计算
        roc_auc = auc(fpr, tpr)
        # 绘制ROC曲线
        plt.plot(fpr, tpr, linestyle='-', label=mode + ' ROC (AUC = {0:.4f})'.format(roc_auc), color="black",
                 linewidth=1)
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = mode + " ROC Curve"
        plt.title(title)
        plt.legend(loc="best")
        plt.savefig(fname=title + ".svg", dpi=2000, format="svg")
        plt.show()

        # 绘制混淆矩阵
        bool_2_string = np.array(["neg", "pos"])
        predict_label = bool_2_string[(y_pred > 0).astype(np.int32)]
        true_label = bool_2_string[y_true]
        cm = confusion_matrix(true_label, predict_label, labels=["neg", "pos"])
        df = pd.DataFrame(cm, index=["neg", "pos"], columns=["neg", "pos"])
        seaborn.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        title = mode + " Confusion Matrix Graph"
        plt.title(title)
        plt.savefig(fname=title + ".svg", dpi=2000, format="svg")
        plt.show()


# 对比分析满足和不满足朴素贝叶斯假设下的分类精度
def hypothesis_test(n_iters=10):
    n_train = 200
    n_test = 2000
    pos_mean_vector = np.array([1.0, 1.6])
    neg_mean_vector = np.array([-0.5, 0.0])
    covariance_matrix_independence = np.array([[0.4, 0.0], [0.0, 0.6]])
    covariance_matrix_not_independence = np.array([[0.4, 0.1], [0.1, 0.6]])
    # satisfy conditional independence case:
    test_acc_independence = 0
    for i in range(n_iters):
        X_train, y_train, train_pos_samples_num, \
        train_neg_samples_num = generate_data(n_samples=n_train,
                                              pos_mean_vector=pos_mean_vector,
                                              neg_mean_vector=neg_mean_vector,
                                              satisfy_naive_bayesian_hypothesis=True,
                                              covariance_matrix=covariance_matrix_independence)
        X_test, y_test, test_pos_samples_num, \
        test_neg_samples_num = generate_data(n_samples=n_test,
                                             pos_mean_vector=pos_mean_vector,
                                             neg_mean_vector=neg_mean_vector,
                                             satisfy_naive_bayesian_hypothesis=True,
                                             covariance_matrix=covariance_matrix_independence)
        logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                                test_pos_samples_num=test_pos_samples_num,
                                                train_neg_samples_num=train_neg_samples_num,
                                                test_neg_samples_num=test_neg_samples_num,
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                regular_coef=0.0, verbose=False)
        w, train_acc, test_acc, actual_iter_times, train_loss_list = logistic.train(train_method="gradient descent",
                                                                                    train_param=[1, 20000, 1e-6],
                                                                                    draw_result=False)
        test_acc_independence += test_acc
    test_acc_independence /= n_iters

    # not satisfy conditional independence case:
    test_acc_not_independence = 0
    for i in range(n_iters):
        X_train, y_train, train_pos_samples_num, \
        train_neg_samples_num = generate_data(n_samples=n_train,
                                              pos_mean_vector=pos_mean_vector,
                                              neg_mean_vector=neg_mean_vector,
                                              satisfy_naive_bayesian_hypothesis=False,
                                              covariance_matrix=covariance_matrix_not_independence)
        X_test, y_test, test_pos_samples_num, \
        test_neg_samples_num = generate_data(n_samples=n_test,
                                             pos_mean_vector=pos_mean_vector,
                                             neg_mean_vector=neg_mean_vector,
                                             satisfy_naive_bayesian_hypothesis=False,
                                             covariance_matrix=covariance_matrix_not_independence)
        logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                                test_pos_samples_num=test_pos_samples_num,
                                                train_neg_samples_num=train_neg_samples_num,
                                                test_neg_samples_num=test_neg_samples_num,
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                regular_coef=0.0, verbose=False)
        w, train_acc, test_acc, actual_iter_times, train_loss_list = logistic.train(train_method="gradient descent",
                                                                                    train_param=[1, 20000, 1e-6],
                                                                                    draw_result=False)
        test_acc_not_independence += test_acc
    test_acc_not_independence /= n_iters

    print("iter " + str(n_iters) + " times finished!")
    print("test_acc when satisfy conditional independence:", round(test_acc_independence, 4))
    print("test_acc when not satisfy conditional independence:", round(test_acc_not_independence, 4))


# 绘制accuracy随l2_coefficient的变化图
def draw_accuracy_regular_coef_graph(training_times: int = 10, train_method="newton", train_param=None):
    assert training_times > 0
    if is_show:
        mean_train_acc_list = []
        mean_test_acc_list = []
        regular_coef_range = np.arange(0.01, 0.51, 0.01)
        n_train = 20
        n_test = 2000
        pos_mean_vector = np.array([1.0, 1.6])
        neg_mean_vector = np.array([-0.5, 0.0])
        covariance_matrix = np.array([[0.4, 0.0], [0.0, 0.6]])

        for regular_coef in regular_coef_range:
            mean_train_acc = 0
            mean_test_acc = 0
            for _ in range(training_times):
                X_train, y_train, train_pos_samples_num, \
                train_neg_samples_num = generate_data(n_samples=n_train,
                                                      pos_mean_vector=pos_mean_vector,
                                                      neg_mean_vector=neg_mean_vector,
                                                      satisfy_naive_bayesian_hypothesis=True,
                                                      covariance_matrix=covariance_matrix)
                X_test, y_test, test_pos_samples_num, \
                test_neg_samples_num = generate_data(n_samples=n_test,
                                                     pos_mean_vector=pos_mean_vector,
                                                     neg_mean_vector=neg_mean_vector,
                                                     satisfy_naive_bayesian_hypothesis=True,
                                                     covariance_matrix=covariance_matrix)

                logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                                        test_pos_samples_num=test_pos_samples_num,
                                                        train_neg_samples_num=train_neg_samples_num,
                                                        test_neg_samples_num=test_neg_samples_num,
                                                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                        regular_coef=regular_coef, verbose=False)
                w, train_acc, test_acc, actual_iter_times, train_loss_list = None, None, None, None, None
                if train_method == "gradient descent":
                    w, train_acc, test_acc, actual_iter_times, train_loss_list = logistic.train(
                        train_method="gradient descent", train_param=[1, 20000, 1e-6], draw_result=False)
                elif train_method == "newton":
                    w, train_acc, test_acc, actual_iter_times, train_loss_list = logistic.train(
                        train_method="newton", train_param=[50, 1e-6], draw_result=False)
                else:
                    raise NotImplementedError
                mean_train_acc += train_acc
                mean_test_acc += test_acc
            mean_train_acc_list.append(mean_train_acc / training_times)
            mean_test_acc_list.append(mean_test_acc / training_times)
        draw_x = regular_coef_range
        plt.scatter(draw_x, np.array(mean_train_acc_list), marker='o', color='blue', s=10)
        plt.scatter(draw_x, np.array(mean_test_acc_list), marker='o', color='red', s=10)
        plt.plot(draw_x, np.array(mean_train_acc_list), color='blue', linewidth=1.0, linestyle='--',
                 label="train acc")
        plt.plot(draw_x, np.array(mean_test_acc_list), color='red', linewidth=1.0, linestyle='-',
                 label="test acc")
        plt.xlabel("regular_coef")
        plt.ylabel("acc")
        plt.legend(loc="best")
        plt.title("acc_regular_coef graph(" + train_method + ")")
        plt.savefig(fname="acc_regular_coef.svg", dpi=2000, format="svg")
        plt.show()


# 对比加正则项后与加正则项前的拟合效果
def show_compare_regular():
    if is_show:
        n_train = 20
        n_test = 2000
        pos_mean_vector = np.array([1.0, 1.6])
        neg_mean_vector = np.array([-0.5, 0.0])
        hypothesis = True
        covariance_matrix = np.array([[0.4, 0.0], [0.0, 0.6]])
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
        # no regular
        regular_coef = 0.0
        logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                                test_pos_samples_num=test_pos_samples_num,
                                                train_neg_samples_num=train_neg_samples_num,
                                                test_neg_samples_num=test_neg_samples_num,
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                regular_coef=regular_coef, verbose=True)
        w_no_regular, train_acc_no_regular, test_acc_no_regular, actual_iter_times_no_regular, \
        train_loss_list_no_regular = logistic.train(
            train_method="gradient descent",
            train_param=[1, 20000, 1e-6],
            draw_result=True)
        train_y_pred_no_regular = logistic.predict(w_no_regular, X_train)
        test_y_pred_no_regular = logistic.predict(w_no_regular, X_test)
        analysis(y_train, train_y_pred_no_regular, mode="Gradient Descent,Train")
        analysis(y_test, test_y_pred_no_regular, mode="Gradient Descent,Test")

        # regular
        regular_coef = 0.2
        logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                                test_pos_samples_num=test_pos_samples_num,
                                                train_neg_samples_num=train_neg_samples_num,
                                                test_neg_samples_num=test_neg_samples_num,
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                regular_coef=regular_coef, verbose=True)
        w_regular, train_acc_regular, test_acc_regular, actual_iter_times_regular, \
        train_loss_list_regular = logistic.train(
            train_method="gradient descent",
            train_param=[1, 20000, 1e-6],
            draw_result=True)
        train_y_pred_regular = logistic.predict(w_regular, X_train)
        test_y_pred_regular = logistic.predict(w_regular, X_test)
        analysis(y_train, train_y_pred_regular, mode="Gradient Descent,Train")
        analysis(y_test, test_y_pred_regular, mode="Gradient Descent,Test")


# 对比不同方法的迭代次数
def show_compare_method_iter_times():
    if is_show:
        n_train_list = [20, 50, 100, 200, 300, 400, 500, 750, 1000]
        n_test = 2000
        pos_mean_vector = np.array([1.0, 1.6])
        neg_mean_vector = np.array([-0.5, 0.0])
        hypothesis = False
        # hypothesis = True
        covariance_matrix = np.array([[0.4, 0.1], [0.1, 0.6]])
        # covariance_matrix = np.array([[0.4, 0.0], [0.0, 0.6]])
        gd_actual_iter_times_list = []
        newton_actual_iter_times_list = []
        for n_train in n_train_list:
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
            regular_coef = 0.5
            # regular_coef = 0.0
            logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                                    test_pos_samples_num=test_pos_samples_num,
                                                    train_neg_samples_num=train_neg_samples_num,
                                                    test_neg_samples_num=test_neg_samples_num,
                                                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                    regular_coef=regular_coef, verbose=True)
            w_gd, train_acc_gd, test_acc_gd, actual_iter_times_gd, train_loss_list_gd = logistic.train(
                train_method="gradient descent",
                train_param=[1, 20000, 1e-6],
                draw_result=False)
            gd_actual_iter_times_list.append(actual_iter_times_gd)

            w_newton, train_acc_newton, test_acc_newton, actual_iter_times_newton, train_loss_list_newton = logistic.train(
                train_method="newton",
                train_param=[50, 1e-6],
                draw_result=False)
            newton_actual_iter_times_list.append(actual_iter_times_newton)

        draw_x = np.log10(np.array(n_train_list))
        plt.scatter(draw_x, np.array(gd_actual_iter_times_list), marker='o', color='blue', s=10)
        plt.scatter(draw_x, np.array(newton_actual_iter_times_list), marker='o', color='red', s=10)
        plt.plot(draw_x, np.array(gd_actual_iter_times_list), color='blue', linewidth=1.0, linestyle='--',
                 label="gradient descent")
        plt.plot(draw_x, np.array(newton_actual_iter_times_list), color='red', linewidth=1.0, linestyle='-',
                 label="newton")
        plt.xlabel("log10(n_train)")
        plt.ylabel("iter times")
        plt.legend(loc="best")
        plt.title("iter_times_log_n_train_graph")
        plt.savefig(fname="iter_times_log_n_train_graph.svg", dpi=2000, format="svg")
        plt.show()


if __name__ == '__main__':
    # find_best_lr()
    # os.system("pause")

    # hypothesis_test(100)
    # os.system("pause")

    # draw_accuracy_regular_coef_graph()
    # os.system("pause")

    # show_compare_regular()
    # os.system("pause")

    # show_compare_method_iter_times()
    # os.system("pause")

    n_train = 200
    # n_train = 200
    n_test = 2000
    pos_mean_vector = np.array([1.0, 1.6])
    neg_mean_vector = np.array([-0.5, 0.0])
    # hypothesis = False
    hypothesis = True
    # covariance_matrix = np.array([[0.4, 0.1], [0.1, 0.6]])
    covariance_matrix = np.array([[0.4, 0.0], [0.0, 0.6]])
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

    regular_coef = 0.0
    logistic = LG.Logistic_Regression_Class(n_feature=2, train_pos_samples_num=train_pos_samples_num,
                                            test_pos_samples_num=test_pos_samples_num,
                                            train_neg_samples_num=train_neg_samples_num,
                                            test_neg_samples_num=test_neg_samples_num,
                                            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                            regular_coef=regular_coef, verbose=True)
    w_nt, train_acc_nt, test_acc_nt, actual_iter_times_nt, train_loss_list_nt = logistic.train(train_method="newton",
                                                                                               train_param=[50, 1e-6],
                                                                                               draw_result=True)
    w_gd, train_acc_gd, test_acc_gd, actual_iter_times_gd, train_loss_list_gd = logistic.train(
        train_method="gradient descent",
        train_param=[1, 20000, 1e-6],
        draw_result=True)

    train_y_pred_nt = logistic.predict(w_nt, X_train)
    test_y_pred_nt = logistic.predict(w_nt, X_test)
    analysis(y_train, train_y_pred_nt, mode="Newton Method,Train")
    analysis(y_test, test_y_pred_nt, mode="Newton Method,Test")

    train_y_pred_gd = logistic.predict(w_gd, X_train)
    test_y_pred_gd = logistic.predict(w_gd, X_test)
    analysis(y_train, train_y_pred_gd, mode="Gradient Descent,Train")
    analysis(y_test, test_y_pred_gd, mode="Gradient Descent,Test")

    os.system("pause")
    # ===========================================================================================
    # 使用breast_cancer数据集进行二分类逻辑回归实验
    breast_cancer_dataset = datasets.load_breast_cancer()
    print(breast_cancer_dataset.DESCR)
    os.system("pause")
    cancer_X = breast_cancer_dataset.data

    print("cancer_X info:")
    print(pd.DataFrame(cancer_X).info())
    cancer_y = breast_cancer_dataset.target

    print("-------------------------------------------")
    print("cancer_y info:")
    print(pd.DataFrame(cancer_y).info())
    print("-------------------------------------------")


    def sort_by_label(X, y):
        assert len(X) == len(y)
        X_pos, X_neg = [], []
        y_pos, y_neg = [], []
        pos_num, neg_num = 0, 0
        for i in range(len(y)):
            if y[i] == 1:
                pos_num += 1
                X_pos.append(X[i, :])
                y_pos.append(1)
            elif y[i] == 0:
                neg_num += 1
                X_neg.append(X[i, :])
                y_neg.append(0)
            else:
                assert False
        sorted_X = np.array(X_pos + X_neg)
        sorted_y = np.array(y_pos + y_neg)
        print("sorted_X.shape:", sorted_X.shape)
        print("sorted_y.shape:", sorted_y.shape)
        assert len(sorted_X) == len(sorted_y) == neg_num + pos_num == len(y)
        return sorted_X, sorted_y, pos_num, neg_num


    X_train, X_test, y_train, y_test = train_test_split(cancer_X, cancer_y, test_size=0.3)
    X_train, y_train, train_pos_samples_num, train_neg_samples_num = sort_by_label(X_train, y_train)
    X_test, y_test, test_pos_samples_num, test_neg_samples_num = sort_by_label(X_test, y_test)
    n_train, n_feature = X_train.shape
    n_test = len(X_test)
    print("n_train:", n_train, "n_test:", n_test, "n_feature:", n_feature)

    logistic = LG.Logistic_Regression_Class(n_feature=n_feature, train_pos_samples_num=train_pos_samples_num,
                                            test_pos_samples_num=test_pos_samples_num,
                                            train_neg_samples_num=train_neg_samples_num,
                                            test_neg_samples_num=test_neg_samples_num,
                                            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                            regular_coef=0.15, verbose=True)

    w, train_acc, test_acc, actual_iter_times, train_loss_list = logistic.train(train_method="newton",
                                                                                train_param=[50, 1e-6],
                                                                                draw_result=False)
    train_y_pred = logistic.predict(w, X_train)
    test_y_pred = logistic.predict(w, X_test)
    analysis(y_train, train_y_pred, mode="Train")
    analysis(y_test, test_y_pred, mode="Test")
    print("finish")
