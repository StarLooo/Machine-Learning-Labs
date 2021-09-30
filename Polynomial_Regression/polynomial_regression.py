# -*- coding: UTF-8 -*-
from abc import ABC

import numpy as np

from utils import *
import abc


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, train_param, hyper_params, verbose=False):
        self.train_param = train_param
        self.hyper_params = hyper_params
        self.verbose = verbose

    @abc.abstractmethod
    def train(self):
        pass


class Analytic_Optimizer(Optimizer, ABC):
    def __init__(self, train_param, hyper_params, loss_func, analytic_func, verbose=False):
        super().__init__(train_param, hyper_params, verbose)
        self.loss_func = loss_func
        self.analytic_func = analytic_func

    def train(self):
        if self.verbose:
            print("optimize with analytic method.")
        self.train_param = self.analytic_func()
        train_loss = self.loss_func(self.train_param)
        if self.verbose:
            print("analytic solve finished.")
        return self.train_param, train_loss


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
            new_param = self.train_param - latest_grad * self.train_param  # 梯度下降
            train_loss = self.loss_func(self.train_param)  # 计算本次迭代后的训练误差
            # 若loss不再下降，则不更新参数，并减半学习率
            if train_loss >= pre_loss:
                self.lr /= 2  # 减小学习率
                train_loss = pre_loss
            # 否则更新参数
            self.train_param = new_param

            # 若loss下降小于阈值，则训练结束，退出循环
            if pre_loss - train_loss < self.epsilon:
                if self.verbose:
                    print("gradient descent finished, actual iter times:", actual_iter_times)
                break
            # 若迭代次数达到上限，训练结束
            if actual_iter_times == self.max_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list, latest_grad


class Conjugate_Gradient_Optimizer(Optimizer, ABC):
    def __init__(self, train_param, hyper_params, A, b, loss_func, verbose=False):
        super().__init__(train_param, hyper_params, verbose)
        self.max_iter_times, self.epsilon = hyper_params
        assert self.max_iter_times > 0 and self.epsilon >= 0.
        assert A.shape[0] == b.shape[0]
        self.A = A
        self.b = b
        self.loss_func = loss_func

    def train(self):
        if self.verbose:
            print("optimize with conjugate gradient.")
        train_loss_list = []
        train_loss = self.loss_func(self.train_param)
        actual_iter_times = 0

        r = self.b - np.matmul(self.A, self.train_param)
        p = r
        for iter_num in range(1, self.max_iter_times + 1):
            actual_iter_times = iter_num
            pre_loss = train_loss  # 上一次迭代的loss
            train_loss_list.append(train_loss)  # 记录train_loss
            pre_param = self.train_param  # 上一次迭代的w
            old_r_inner_product = np.matmul(r.transpose(), r)
            alpha = old_r_inner_product / np.matmul(np.matmul(p.transpose(), self.A), p)
            self.train_param = self.train_param + alpha * p
            r = r - alpha * np.matmul(self.A, p)
            new_r_inner_product = np.matmul(r.transpose(), r)
            beta = new_r_inner_product / old_r_inner_product
            p = r + beta * p

            # 计算本次迭代后的训练误差
            train_loss = self.loss_func()
            assert train_loss < pre_loss
            # 残差r的L1-norm小于阈值，则训练结束，退出循环
            if np.max(np.abs(r)) < self.epsilon:
                if self.verbose:
                    print("conjugate gradient finished, actual iter times:", actual_iter_times)
                break
            if iter_num == actual_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list


class Polynomial_Regression_Class:
    def __init__(self, m: int, n_train: int = 10, n_test: int = 990, l2_norm_coefficient: float = 0., verbose=True,
                 train_data=None, test_data=None):
        assert m > 0 and n_train > 0 and n_test > 0 and l2_norm_coefficient >= 0.
        # self.DEBUG = False
        self.m = m
        self.n_train = n_train
        self.n_test = n_test
        self.verbose = verbose
        self.sigma = 0.2
        if train_data is None:
            self.train_x_array, self.train_y_array = generate_data(n_train, self.sigma)
        else:
            self.train_x_array, self.train_y_array = train_data
        if test_data is None:
            self.test_x_array, self.test_y_array = generate_data(n_test, self.sigma)
        else:
            self.test_x_array, self.test_y_array = test_data
        self.model_name = "polynomial regression with order of " + str(m)
        self.l2_norm_coefficient = l2_norm_coefficient
        if l2_norm_coefficient == 0.:
            self.model_name += ", using L2-norm regular, l2 norm coefficient = " + str(l2_norm_coefficient) + "."
        else:
            self.model_name += ", without regular."
        if self.verbose:
            print("initial " + self.model_name)
            print("n_train, n_test:", n_train, n_test)

    def _mse_loss_with_l2_norm_regular(self, y_pred, y_true, w):
        assert len(y_pred) == len(y_true)
        mse_loss = (np.sum(np.square(y_pred - y_true))) / 2
        l2_norm_regular = (self.l2_norm_coefficient / 2) * (np.sum(np.square(w)))
        return mse_loss + l2_norm_regular

    def _rmse(self, y_pred, y_true, w):
        return np.sqrt((2 / len(y_pred)) * self._mse_loss_with_l2_norm_regular(y_pred, y_true, w))

    def _solve_analytic(self, draw_result: bool = False):
        w = np.empty(self.m + 1)
        X = np.vander(self.train_x_array, self.m + 1)
        X_t = X.transpose()
        X_t_mul_X = np.matmul(X_t, X)
        A = X_t_mul_X + self.l2_norm_coefficient * np.eye(self.m + 1)
        A_pinv = np.linalg.pinv(A)
        b = np.matmul(X_t, self.train_y_array)

        # 传入Analytic_Optimizer的解析法求解函数
        def analytic_func():
            return np.matmul(A_pinv, b)

        # 传入Analytic_Optimizer的train_loss计算函数
        def loss_func(w):
            y_pred_train = np.matmul(X, w)
            loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
            return loss

        analytic_opt = Analytic_Optimizer(w, None, loss_func, analytic_func, self.verbose)  # 初始化Analytic_Optimizer
        w, train_loss = analytic_opt.train()  # 使用解析法求出w的解析解以及train_loss

        # 计算测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("analytic, train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))

        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1), w)
            title = "n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse

    def _solve_gradient_descent(self, lr: float, max_iters: int, epsilon: float, draw_result: bool = False):
        assert lr > 0. and max_iters > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.m + 1)
        # 梯度下降法求解w
        X = np.vander(self.train_x_array, self.m + 1)
        X_t = X.transpose()
        X_t_mul_X = np.matmul(X_t, X)
        y_pred_train = np.matmul(X, w)
        train_loss_list = []
        train_loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
        iter_times = 0
        for iter_num in range(1, max_iters + 1):
            iter_times = iter_num
            pre_loss = train_loss  # 上一次迭代的loss
            train_loss_list.append(train_loss)
            pre_w = w  # 上一次迭代的w
            gradient_w = np.matmul(X_t, np.matmul(X, w) - self.train_y_array) + \
                         self.l2_norm_coefficient * w  # 求梯度
            w -= lr * gradient_w  # 梯度下降
            # 计算本次迭代后训练集上新的预测值和训练误差
            y_pred_train = np.matmul(X, w)
            train_loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
            # loss不再下降，考虑减小学习率
            if train_loss >= pre_loss:
                w = pre_w  # 还原错误更新的参数
                lr /= 2  # 减小学习率
            # loss下降小于阈值，则退出循环
            elif epsilon != 0. and pre_loss - train_loss < epsilon:
                if self.verbose:
                    print("gradient descent finished, iteration times:", iter_times)
                break
            if iter_num == max_iters:
                if self.verbose:
                    print("iter too many times, terminate train!")
            train_loss_list.append(train_loss)

        # 计算训练集和测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))

        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1), w)
            title = "gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse, iter_times, train_loss_list

    def _solve_conjugate_gradient_descent(self, max_iters: int, epsilon: float, draw_result: bool = False):
        assert max_iters > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.m + 1)
        # 共轭梯度下降法求解w
        X = np.vander(self.train_x_array, self.m + 1)
        X_t = X.transpose()
        X_t_mul_X = np.matmul(X_t, X)
        y_pred_train = np.matmul(X, w)
        A = X_t_mul_X + self.l2_norm_coefficient * np.eye(self.m + 1)
        b = np.matmul(X_t, self.train_y_array)
        train_loss_list = []
        train_loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
        iter_times = 0
        r = b - np.matmul(A, w)
        p = r

        for iter_num in range(1, max_iters + 1):
            iter_times = iter_num
            pre_loss = train_loss  # 上一次迭代的loss
            train_loss_list.append(train_loss)
            pre_w = w  # 上一次迭代的w
            old_r_inner_product = np.matmul(r.transpose(), r)
            alpha = old_r_inner_product / np.matmul(np.matmul(p.transpose(), A), p)
            w = w + alpha * p
            r = r - alpha * np.matmul(A, p)
            new_r_inner_product = np.matmul(r.transpose(), r)
            beta = new_r_inner_product / old_r_inner_product
            p = r + beta * p

            # 计算本次迭代后训练集上新的预测值和训练误差
            y_pred_train = np.matmul(X, w)
            train_loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)

            assert train_loss <= pre_loss
            # loss下降小于阈值，则退出循环
            if epsilon != 0. and np.max(np.abs(r)) < epsilon:
                if self.verbose:
                    print("gradient descent finished, iteration times:", iter_times)
                break
            if iter_num == max_iters:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        # 计算训练集和测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))

        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1), w)
            title = "conjugate gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse, iter_times, train_loss_list

    def train(self, train_method="analytic", train_param=None, draw_result=False):
        # 使用解析法求解
        if train_method == "analytic":
            if self.verbose:
                print("train the model with analytic method:")
            return self._solve_analytic(draw_result)
        # 使用梯度下降法
        elif train_method == "gradient descent":
            if self.verbose:
                print("train the model with gradient descent:")
            lr, max_iters, epsilon = train_param
            return self._solve_gradient_descent(lr, max_iters, epsilon, draw_result)
        # 使用共轭梯度下降法
        elif train_method == "conjugate gradient descent":
            if self.verbose:
                print("train the model with conjugate gradient descent:")
            max_iters, epsilon = train_param
            return self._solve_conjugate_gradient_descent(max_iters, epsilon, draw_result)
        # 其他(暂未实现)
        else:
            if self.verbose:
                print("this method hasn't implemented!:")
            raise NotImplementedError
