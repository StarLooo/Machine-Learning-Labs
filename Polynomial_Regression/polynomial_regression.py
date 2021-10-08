# -*- coding: UTF-8 -*-
import scipy.linalg as linalg
from utils import *


class Polynomial_Regression_Class:
    # 初始化
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

    # 平方损失
    def _mse_loss_with_l2_norm_regular(self, y_pred, y_true, w):
        assert len(y_pred) == len(y_true)
        mse_loss = (np.sum(np.square(y_pred - y_true))) / 2
        if self.l2_norm_coefficient > 0.:
            l2_norm_regular = (self.l2_norm_coefficient / 2) * (np.matmul(w.transpose(), w))
        else:
            l2_norm_regular = 0
        return mse_loss + l2_norm_regular

    # 均方根损失
    def _rmse(self, y_pred, y_true, w):
        return np.sqrt((2 / len(y_pred)) * self._mse_loss_with_l2_norm_regular(y_pred, y_true, w))

    # 解析法求解
    def _solve_analytic(self, draw_result: bool = False):
        w = np.empty(self.m + 1)
        X = np.vander(self.train_x_array, self.m + 1, increasing=True)
        X_t = X.transpose()
        X_t_mul_X = np.matmul(X_t, X)
        A = X_t_mul_X + self.l2_norm_coefficient * np.eye(self.m + 1)
        A_pinv = linalg.pinv(A)
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
        w, train_loss = analytic_opt.train()  # 使用解析法求出w的解析解

        # 计算测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("analytic, train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))
            print("w:", w)
            print("final train_loss:", train_loss)
        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
            title = "analytic, n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", l2_coef=" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse

    # 梯度下降法求解
    def _solve_gradient_descent(self, lr: float, max_iter_times: int, epsilon: float, draw_result: bool = False):
        assert lr > 0. and max_iter_times > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.m + 1)

        # 梯度下降法求解w
        X = np.vander(self.train_x_array, self.m + 1, increasing=True)
        X_t = X.transpose()

        # 传入Gradient_Descent_Optimizer的梯度函数
        def grad_func(w):
            return np.matmul(X_t, np.matmul(X, w) - self.train_y_array) + self.l2_norm_coefficient * w

        # 传入Gradient_Descent_Optimizer的train_loss计算函数
        def loss_func(w):
            y_pred_train = np.matmul(X, w)
            loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
            return loss

        gd_opt = Gradient_Descent_Optimizer(w, [lr, max_iter_times, epsilon], loss_func, grad_func,
                                            self.verbose)  # 初始化Gradient_Descent_Optimizer
        w, actual_iter_times, train_loss_list, latest_grad = gd_opt.train()  # 使用梯度下降法求出w的解
        if self.verbose:
            print("L1-norm of latest gradient:", np.max(latest_grad))
            print("w:", w)
            print("train_loss:", train_loss_list[-1])
            print("actual iter times:", actual_iter_times)

        # 计算训练集和测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))

        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
            title = "gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", l2_coef=" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse, actual_iter_times, train_loss_list

    # 精确线搜索的梯度下降
    def _solve_els_gradient_descent(self, max_iter_times: int, epsilon: float, draw_result: bool = False):
        assert max_iter_times > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.m + 1)

        # 梯度下降法求解w
        X = np.vander(self.train_x_array, self.m + 1, increasing=True)
        X_t = X.transpose()
        X_t_mul_X = np.matmul(X_t, X)

        # 传入ELS_Gradient_Descent_Optimizer的梯度函数
        def grad_func(w):
            return np.matmul(X_t, np.matmul(X, w) - self.train_y_array) + self.l2_norm_coefficient * w

        # 传入ELS_Gradient_Descent_Optimizer的train_loss计算函数
        def loss_func(w):
            y_pred_train = np.matmul(X, w)
            loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
            return loss

        # 传入ELS_Gradient_Descent_Optimizer的精确线搜索函数
        def els_func(grad):
            step_len = np.dot(grad, grad) / np.dot(np.matmul(X_t_mul_X, grad), np.matmul(X_t_mul_X, grad))
            return step_len

        gd_opt = ELS_Gradient_Descent_Optimizer(w, [max_iter_times, epsilon], loss_func, grad_func, els_func,
                                                self.verbose)  # 初始化ELS_Gradient_Descent_Optimizer
        w, actual_iter_times, train_loss_list, latest_grad = gd_opt.train()  # 使用精确线搜索的梯度下降法求出w的解
        if self.verbose:
            print("L1-norm of latest gradient:", np.max(latest_grad))
            print("w:", w)
            print("train_loss:", train_loss_list[-1])

        # 计算训练集和测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))

        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
            title = "gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse, actual_iter_times, train_loss_list

    # 随机梯度下降求解
    def _solve_stochastic_gradient_descent(self, lr: float, max_iter_times: int, epsilon: float,
                                           draw_result: bool = False):
        assert lr > 0. and max_iter_times > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.m + 1)

        # 梯度下降法求解w
        X = np.vander(self.train_x_array, self.m + 1, increasing=True)
        X_t = X.transpose()

        # 传入Gradient_Descent_Optimizer的梯度函数
        def grad_func(w, i):
            x = X[i]
            grad = (np.dot(x, w) - self.train_y_array[i]) * x
            assert grad.shape == (self.m + 1,)
            return grad

        # 传入Gradient_Descent_Optimizer的train_loss计算函数
        def loss_func(w):
            y_pred_train = np.matmul(X, w)
            loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
            return loss

        gd_opt = Stochastic_Gradient_Descent_Optimizer(w, [lr, max_iter_times, epsilon], self.n_train, loss_func,
                                                       grad_func, self.verbose)  # 初始化Analytic_Optimizer
        w, actual_iter_times, train_loss_list = gd_opt.train()  # 使用解析法求出w的解析解以及train_loss
        if self.verbose:
            print("w:", w)
            print("train_loss:", train_loss_list[-1])

        # 计算训练集和测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))

        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
            title = "stochastic gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse, actual_iter_times, train_loss_list

    # 共轭梯度法求解
    def _solve_conjugate_gradient_descent(self, max_iter_times: int, epsilon: float, draw_result: bool = False):
        assert max_iter_times > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.m + 1)
        w = np.empty(self.m + 1)
        X = np.vander(self.train_x_array, self.m + 1, increasing=True)
        X_t = X.transpose()
        X_t_mul_X = np.matmul(X_t, X)
        A = X_t_mul_X + self.l2_norm_coefficient * np.eye(self.m + 1)
        b = np.matmul(X_t, self.train_y_array)

        # 传入Conjugate_Gradient_Optimizer的train_loss计算函数
        def loss_func(w):
            y_pred_train = np.matmul(X, w)
            loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
            return loss

        cg_opt = Conjugate_Gradient_Optimizer(w, [max_iter_times, epsilon], A, b, loss_func,
                                              self.verbose)  # 初始化Conjugate_Gradient_Optimizer
        w, actual_iter_times, train_loss_list = cg_opt.train()  # 使用共轭梯度法求出w

        # 计算训练集和测试集上的预测值
        y_pred_train = np.matmul(X, w)
        y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
        # 计算训练集和测试集上的均方根误差
        train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
        test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
        if self.verbose:
            print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))

        # 画图分析结果
        if draw_result:
            draw_x = np.linspace(0, 1, 200)
            draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
            title = "conjugate gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
            if self.l2_norm_coefficient > 0.0:
                title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
            draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)

        return w, train_rmse, test_rmse, actual_iter_times, train_loss_list

    # 训练
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
        # 使用精确线搜索的梯度下降法
        elif train_method == "els gradient descent":
            if self.verbose:
                print("train the model with exact line search gradient descent:")
            max_iters, epsilon = train_param
            return self._solve_els_gradient_descent(max_iters, epsilon, draw_result)
        # 使用随机梯度下降法
        elif train_method == "stochastic gradient descent":
            if self.verbose:
                print("train the model with stochastic gradient descent:")
            lr, max_iters, epsilon = train_param
            return self._solve_stochastic_gradient_descent(lr, max_iters, epsilon, draw_result)
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
