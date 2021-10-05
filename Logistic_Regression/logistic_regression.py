# -*- coding: UTF-8 -*-

from utils import *


class Logistic_Regression_Class:
    # 初始化
    def __init__(self, n_feature, train_pos_samples_num, test_pos_samples_num, train_neg_samples_num,
                 test_neg_samples_num, X_train, y_train, X_test, y_test, regular_coef=0.0, verbose=False):
        assert n_feature > 0 and train_pos_samples_num > 0 and test_pos_samples_num > 0 \
               and train_neg_samples_num > 0 and test_neg_samples_num > 0
        self.n_feature = n_feature
        self.train_pos_samples_num = train_pos_samples_num
        self.test_pos_samples_num = test_pos_samples_num
        self.train_neg_samples_num = train_neg_samples_num
        self.test_neg_samples_num = test_neg_samples_num
        self.n_train = train_pos_samples_num + train_neg_samples_num
        self.n_test = test_pos_samples_num + test_neg_samples_num

        assert X_train.shape == (self.n_train, self.n_feature) and X_test.shape == (self.n_test, self.n_feature)
        assert y_train.shape == (self.n_train,) and y_test.shape == (self.n_test,)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.verbose = verbose
        self.regular_coef = regular_coef
        self.model_name = "logistic regression with n_feature of " + str(self.n_feature)
        if regular_coef == 0.:
            self.model_name += ", using L2-norm regular, l2 norm coefficient = " + str(regular_coef) + "."
        else:
            self.model_name += ", without regular."
        if self.verbose:
            print("initial " + self.model_name + " finished.")
            print("n_train, n_test:", self.n_train, self.n_test)

    # 精度计算
    def _accuracy(self, w, X, y_true):
        assert len(X) == len(y_true)
        extend_X = np.concatenate((X, np.ones(shape=(len(X), 1))), axis=1)
        y_pred = np.array([np.dot(w, X[i, :]) for i in len(X)])
        y_pred = y_pred[y_pred > 0].astype(np.int32)
        accuracy = np.sum(y_pred == y_true.astype(np.int32)) / len(y_true)
        assert 0.0 < accuracy < 1.0
        return accuracy

    # 优化目标
    def _optimize_objective_func(self, w, X, y):
        assert len(X) == len(y)
        extend_X = np.concatenate((X, np.ones(shape=(len(X), 1))), axis=1)
        loss = 0
        for i in range(len(X)):
            inner_product = np.dot(w, extend_X[i, :])
            # print("inner_product:", inner_product)
            # 这种处理是防止浮点溢出
            if inner_product > 10:
                loss += -y[i] * inner_product + inner_product
            else:
                loss += -y[i] * inner_product + np.log1p(np.exp(inner_product))
        regular = np.dot(w, w)
        loss += 0.5 * self.regular_coef * regular
        return loss

    # 梯度下降法求解
    def _solve_gradient_descent(self, lr: float, max_iter_times: int, epsilon: float, draw_result: bool = False):
        assert lr > 0. and max_iter_times > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.n_feature + 1)

        # 梯度下降法求解w
        # 传入Gradient_Descent_Optimizer的梯度函数
        def grad_func(w):
            extend_X = np.concatenate((self.X_train, np.ones(shape=(self.n_train, 1))), axis=1)
            grad = np.zeros(self.n_feature + 1)
            for i in range(self.n_train):
                # 注意防止浮点溢出
                inner_product = np.dot(w, extend_X[i, :])
                if inner_product >= 0:
                    power = np.exp(-inner_product)
                    grad -= extend_X[i, :] * (self.y_train[i] - 1 + power / (1 + power))
                else:
                    grad -= extend_X[i, :] * (self.y_train[i] - 1 + 1 / (1 + np.exp(inner_product)))
            return grad

        # 传入Gradient_Descent_Optimizer的train_loss计算函数
        def loss_func(w):
            loss = self._optimize_objective_func(w, self.X_train, self.y_train)
            return loss

        gd_opt = Gradient_Descent_Optimizer(w, [lr, max_iter_times, epsilon], loss_func, grad_func,
                                            self.verbose)  # 初始化Gradient_Descent_Optimizer
        w, actual_iter_times, train_loss_list, latest_grad = gd_opt.train()  # 使用梯度下降法求出w的解

        # 计算训练完成后训练集测试集上的accuracy
        train_acc = self._accuracy(w, self.X_train, self.y_train)
        test_acc = self._accuracy(w, self.X_test, self.y_test)

        if self.verbose:
            print("L1-norm of latest gradient:", np.max(latest_grad))
            print("w:", w)
            print("train_acc:", round(train_acc, 4), "test_loss:", round(test_acc, 4))
            print("actual iter times:", actual_iter_times)

        # 画图分析结果
        if draw_result:
            title = "gradient descent"
            if self.regular_coef > 0.0:
                title += ", regular_coef=" + str(round(self.regular_coef, 4))
            draw_predict_analysis(self.X_train, self.train_pos_samples_num, self.X_test, self.test_pos_samples_num, w,
                                  title)

        return w, train_acc, test_acc, actual_iter_times, train_loss_list

    # # 精确线搜索的梯度下降
    # def _solve_els_gradient_descent(self, max_iter_times: int, epsilon: float, draw_result: bool = False):
    #     assert max_iter_times > 0 and epsilon >= 0.
    #     # 初始化w
    #     w = np.zeros(self.m + 1)
    #
    #     # 梯度下降法求解w
    #     X = np.vander(self.train_x_array, self.m + 1, increasing=True)
    #     X_t = X.transpose()
    #     X_t_mul_X = np.matmul(X_t, X)
    #
    #     # 传入ELS_Gradient_Descent_Optimizer的梯度函数
    #     def grad_func(w):
    #         return np.matmul(X_t, np.matmul(X, w) - self.train_y_array) + self.l2_norm_coefficient * w
    #
    #     # 传入ELS_Gradient_Descent_Optimizer的train_loss计算函数
    #     def loss_func(w):
    #         y_pred_train = np.matmul(X, w)
    #         loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
    #         return loss
    #
    #     # 传入ELS_Gradient_Descent_Optimizer的精确线搜索函数
    #     def els_func(grad):
    #         step_len = np.dot(grad, grad) / np.dot(np.matmul(X_t_mul_X, grad), np.matmul(X_t_mul_X, grad))
    #         return step_len
    #
    #     gd_opt = ELS_Gradient_Descent_Optimizer(w, [max_iter_times, epsilon], loss_func, grad_func, els_func,
    #                                             self.verbose)  # 初始化ELS_Gradient_Descent_Optimizer
    #     w, actual_iter_times, train_loss_list, latest_grad = gd_opt.train()  # 使用精确线搜索的梯度下降法求出w的解
    #     if self.verbose:
    #         print("L1-norm of latest gradient:", np.max(latest_grad))
    #         print("w:", w)
    #         print("train_loss:", train_loss_list[-1])
    #
    #     # 计算训练集和测试集上的预测值
    #     y_pred_train = np.matmul(X, w)
    #     y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
    #     # 计算训练集和测试集上的均方根误差
    #     train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
    #     test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
    #     if self.verbose:
    #         print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))
    #
    #     # 画图分析结果
    #     if draw_result:
    #         draw_x = np.linspace(0, 1, 200)
    #         draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
    #         title = "gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
    #         if self.l2_norm_coefficient > 0.0:
    #             title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
    #         draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)
    #
    #     return w, train_rmse, test_rmse, actual_iter_times, train_loss_list
    #
    # # 随机梯度下降求解
    # def _solve_stochastic_gradient_descent(self, lr: float, max_iter_times: int, epsilon: float,
    #                                        draw_result: bool = False):
    #     assert lr > 0. and max_iter_times > 0 and epsilon >= 0.
    #     # 初始化w
    #     w = np.zeros(self.m + 1)
    #
    #     # 梯度下降法求解w
    #     X = np.vander(self.train_x_array, self.m + 1, increasing=True)
    #     X_t = X.transpose()
    #
    #     # 传入Gradient_Descent_Optimizer的梯度函数
    #     def grad_func(w, i):
    #         x = X[i]
    #         grad = (np.dot(x, w) - self.train_y_array[i]) * x
    #         assert grad.shape == (self.m + 1,)
    #         return grad
    #
    #     # 传入Gradient_Descent_Optimizer的train_loss计算函数
    #     def loss_func(w):
    #         y_pred_train = np.matmul(X, w)
    #         loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
    #         return loss
    #
    #     gd_opt = Stochastic_Gradient_Descent_Optimizer(w, [lr, max_iter_times, epsilon], self.n_train, loss_func,
    #                                                    grad_func, self.verbose)  # 初始化Analytic_Optimizer
    #     w, actual_iter_times, train_loss_list = gd_opt.train()  # 使用解析法求出w的解析解以及train_loss
    #     if self.verbose:
    #         print("w:", w)
    #         print("train_loss:", train_loss_list[-1])
    #
    #     # 计算训练集和测试集上的预测值
    #     y_pred_train = np.matmul(X, w)
    #     y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
    #     # 计算训练集和测试集上的均方根误差
    #     train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
    #     test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
    #     if self.verbose:
    #         print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))
    #
    #     # 画图分析结果
    #     if draw_result:
    #         draw_x = np.linspace(0, 1, 200)
    #         draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
    #         title = "stochastic gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
    #         if self.l2_norm_coefficient > 0.0:
    #             title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
    #         draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)
    #
    #     return w, train_rmse, test_rmse, actual_iter_times, train_loss_list
    #
    # # 共轭梯度法求解
    # def _solve_conjugate_gradient_descent(self, max_iter_times: int, epsilon: float, draw_result: bool = False):
    #     assert max_iter_times > 0 and epsilon >= 0.
    #     # 初始化w
    #     w = np.zeros(self.m + 1)
    #     w = np.empty(self.m + 1)
    #     X = np.vander(self.train_x_array, self.m + 1, increasing=True)
    #     X_t = X.transpose()
    #     X_t_mul_X = np.matmul(X_t, X)
    #     A = X_t_mul_X + self.l2_norm_coefficient * np.eye(self.m + 1)
    #     b = np.matmul(X_t, self.train_y_array)
    #
    #     # 传入Conjugate_Gradient_Optimizer的train_loss计算函数
    #     def loss_func(w):
    #         y_pred_train = np.matmul(X, w)
    #         loss = self._mse_loss_with_l2_norm_regular(y_pred_train, self.train_y_array, w)
    #         return loss
    #
    #     cg_opt = Conjugate_Gradient_Optimizer(w, [max_iter_times, epsilon], A, b, loss_func,
    #                                           self.verbose)  # 初始化Conjugate_Gradient_Optimizer
    #     w, actual_iter_times, train_loss_list = cg_opt.train()  # 使用共轭梯度法求出w
    #
    #     # 计算训练集和测试集上的预测值
    #     y_pred_train = np.matmul(X, w)
    #     y_pred_test = np.matmul(np.vander(self.test_x_array, self.m + 1, increasing=True), w)
    #     # 计算训练集和测试集上的均方根误差
    #     train_rmse = self._rmse(y_pred_train, self.train_y_array, w)
    #     test_rmse = self._rmse(y_pred_test, self.test_y_array, w)
    #     if self.verbose:
    #         print("train rmse:", round(train_rmse, 4), "test rmse:", round(test_rmse, 4))
    #
    #     # 画图分析结果
    #     if draw_result:
    #         draw_x = np.linspace(0, 1, 200)
    #         draw_predict_y = np.matmul(np.vander(draw_x, self.m + 1, increasing=True), w)
    #         title = "conjugate gradient descent, n_train=" + str(self.n_train) + ", m=" + str(self.m)
    #         if self.l2_norm_coefficient > 0.0:
    #             title += ", regular coef:" + str(round(self.l2_norm_coefficient, 4))
    #         draw_predict_analysis(self.train_x_array, self.train_y_array, draw_x, draw_predict_y, title)
    #
    #     return w, train_rmse, test_rmse, actual_iter_times, train_loss_list

    # 训练

    def train(self, train_method, train_param, draw_result=False):
        # 使用梯度下降法
        if train_method == "gradient descent":
            if self.verbose:
                print("train the model with gradient descent:")
            lr, max_iters, epsilon = train_param
            return self._solve_gradient_descent(lr, max_iters, epsilon, draw_result)
        # # 使用精确线搜索的梯度下降法
        # elif train_method == "els gradient descent":
        #     if self.verbose:
        #         print("train the model with exact line search gradient descent:")
        #     max_iters, epsilon = train_param
        #     return self._solve_els_gradient_descent(max_iters, epsilon, draw_result)
        # # 使用随机梯度下降法
        # elif train_method == "stochastic gradient descent":
        #     if self.verbose:
        #         print("train the model with stochastic gradient descent:")
        #     lr, max_iters, epsilon = train_param
        #     return self._solve_stochastic_gradient_descent(lr, max_iters, epsilon, draw_result)
        # # 使用共轭梯度下降法
        # elif train_method == "conjugate gradient descent":
        #     if self.verbose:
        #         print("train the model with conjugate gradient descent:")
        #     max_iters, epsilon = train_param
        #     return self._solve_conjugate_gradient_descent(max_iters, epsilon, draw_result)
        # 其他(暂未实现)
        else:
            if self.verbose:
                print("this method hasn't implemented!:")
            raise NotImplementedError
