# -*- coding: UTF-8 -*-


from Logistic_Regression.utils import *


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
        if regular_coef > 0.:
            self.model_name += ", using L2-norm regular, l2 norm coefficient = " + str(regular_coef) + "."
        else:
            self.model_name += ", without regular."
        if self.verbose:
            print("initial " + self.model_name)
            print("n_train, n_test:", self.n_train, self.n_test)

    # 向量化的sigmoid函数
    def _sigmoid(self, vector_x):
        # 参考网上的经验，用这种方法解决溢出问题
        # 把大于0和小于0的元素分别处理
        # 当vector_x是比较小的负数时会出现上溢，此时可以通过计算exp(vector_x) / (1+exp(vector_x)) 来解决

        mask = (vector_x > 0)
        positive_out = np.zeros_like(vector_x, dtype='float64')
        negative_out = np.zeros_like(vector_x, dtype='float64')

        # 大于0的情况
        positive_out = 1 / (1 + np.exp(-vector_x, positive_out, where=mask))
        # 清除对小于等于0元素的影响
        positive_out[~mask] = 0

        # 小于等于0的情况
        expZ = np.exp(vector_x, negative_out, where=~mask)
        negative_out = expZ / (1 + expZ)
        # 清除对大于0元素的影响
        negative_out[mask] = 0

        return positive_out + negative_out

    # 预测
    def predict(self, w, X):
        extend_X = np.concatenate((np.ones(shape=(len(X), 1)), X), axis=1)
        y_pred = np.array([np.dot(w, extend_X[i, :]) for i in range(len(X))])
        return y_pred

    # 精度计算
    def _accuracy(self, w, X, y_true):
        assert len(X) == len(y_true)
        y_pred = (self.predict(w, X) > 0).astype(int)
        accuracy = np.sum(y_pred == y_true.astype(np.int32)) / len(y_true)
        assert 0.0 <= accuracy <= 1.0
        return accuracy

    # 优化目标
    def _optimize_objective_func(self, w, X, y):
        assert len(X) == len(y)
        extend_X = np.concatenate((np.ones(shape=(len(X), 1)), X), axis=1)
        loss = 0
        for i in range(len(X)):
            inner_product = np.dot(w, extend_X[i, :])
            # print("inner_product:", inner_product)
            # 这种处理是防止浮点溢出
            if inner_product > 15:
                loss += -y[i] * inner_product + inner_product
            else:
                loss += -y[i] * inner_product + np.log1p(np.exp(inner_product))
        regular = np.dot(w, w)
        loss += 0.5 * self.regular_coef * regular
        return loss

    # 牛顿法求解
    def _solve_newton_method(self, max_iter_times: int, epsilon: float, draw_result: bool = False):
        assert max_iter_times > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.n_feature + 1)

        # 牛顿法求解w
        # 传入Newton_Optimizer的一阶导函数
        def first_grad_func(w):
            extend_X = np.concatenate((np.ones(shape=(self.n_train, 1)), self.X_train,), axis=1)
            assert extend_X.shape == (self.n_train, self.n_feature + 1)
            first_grad = np.matmul(extend_X.T,
                                   self._sigmoid(np.matmul(extend_X, w)) - self.y_train) + self.regular_coef * w
            assert first_grad.shape == (self.n_feature + 1,)
            return first_grad

        # 传入Newton_Optimizer的二阶导函数
        def second_grad_func(w):
            extend_X = np.concatenate((np.ones(shape=(self.n_train, 1)), self.X_train), axis=1)
            assert extend_X.shape == (self.n_train, self.n_feature + 1)
            p1 = self._sigmoid(np.matmul(extend_X, w))
            p0 = 1 - p1
            p = p0 * p1
            assert p.shape == p0.shape == p1.shape == (self.n_train,)
            V = np.diag(p)
            second_grad = np.matmul(np.matmul(extend_X.T, V), extend_X) + self.regular_coef * np.eye(self.n_feature + 1)
            assert second_grad.shape == (self.n_feature + 1, self.n_feature + 1)
            return second_grad

        # 传入Gradient_Descent_Optimizer的train_loss计算函数
        def loss_func(w):
            loss = self._optimize_objective_func(w, self.X_train, self.y_train)
            return loss

        newton_opt = Newton_Optimizer(w, [max_iter_times, epsilon], loss_func, first_grad_func, second_grad_func,
                                      self.verbose)  # 初始化Newton_Optimizer
        w, actual_iter_times, train_loss_list, first_grad, second_grad = newton_opt.train()  # 使用牛顿法求出w的解

        # 计算训练完成后训练集测试集上的accuracy
        train_acc = self._accuracy(w, self.X_train, self.y_train)
        test_acc = self._accuracy(w, self.X_test, self.y_test)

        if self.verbose:
            print("L1-norm of latest first gradient:", np.max(first_grad))
            print("w:", w)
            print("train_acc:", round(train_acc, 4), "test_acc:", round(test_acc, 4))
            print("actual iter times:", actual_iter_times)

        # 画图分析结果
        if draw_result:
            title = "newton method"
            if self.regular_coef > 0.0:
                title += ", regular_coef=" + str(round(self.regular_coef, 4))
            draw_predict_analysis(self.X_train, self.train_pos_samples_num, self.X_test, self.test_pos_samples_num, w,
                                  title)

        return w, train_acc, test_acc, actual_iter_times, train_loss_list

    # 梯度下降法求解
    def _solve_gradient_descent(self, lr: float, max_iter_times: int, epsilon: float, draw_result: bool = False):
        assert lr > 0., max_iter_times > 0 and epsilon >= 0.
        # 初始化w
        w = np.zeros(self.n_feature + 1)

        # 梯度下降法求解w
        # 传入Gradient_Descent_Optimizer的梯度函数
        def grad_func(w):
            extend_X = np.concatenate((np.ones(shape=(self.n_train, 1)), self.X_train), axis=1)
            assert extend_X.shape == (self.n_train, self.n_feature + 1)
            # 以下累加式可以向量化，向量化后numpy可以大大提高计算效率
            # grad = np.zeros(self.n_feature + 1)
            # for i in range(self.n_train):
            #     # 注意防止浮点溢出
            #     inner_product = np.dot(w, extend_X[i, :])
            #     if inner_product >= 0:
            #         power = np.exp(-inner_product)
            #         grad -= extend_X[i, :] * (self.y_train[i] - 1 + power / (1 + power))
            #     else:
            #         grad -= extend_X[i, :] * (self.y_train[i] - 1 + 1 / (1 + np.exp(inner_product)))
            # grad += self.regular_coef * w
            grad = np.matmul(extend_X.T, self._sigmoid(np.matmul(extend_X, w)) - self.y_train) + self.regular_coef * w
            assert grad.shape == (self.n_feature + 1,)
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
            print("L1-norm of latest gradient:", np.max(np.abs(latest_grad)))
            print("w:", w)
            print("train_acc:", round(train_acc, 4), "test_acc:", round(test_acc, 4))
            print("actual iter times:", actual_iter_times)

        # 画图分析结果
        if draw_result:
            title = "gradient descent"
            if self.regular_coef > 0.0:
                title += ", regular_coef=" + str(round(self.regular_coef, 4))
            draw_predict_analysis(self.X_train, self.train_pos_samples_num, self.X_test, self.test_pos_samples_num, w,
                                  title)

        return w, train_acc, test_acc, actual_iter_times, train_loss_list

    # 训练
    def train(self, train_method, train_param, draw_result=False):
        # 使用梯度下降法
        if train_method == "gradient descent":
            if self.verbose:
                print("train the model with gradient descent:")
            lr, max_iters, epsilon = train_param
            return self._solve_gradient_descent(lr, max_iters, epsilon, draw_result)
        # 使用牛顿法
        elif train_method == "newton":
            if self.verbose:
                print("train the model with newton method:")
            max_iters, epsilon = train_param
            return self._solve_newton_method(max_iters, epsilon, draw_result)
        # 其他(暂未实现)
        else:
            if self.verbose:
                print("this method hasn't implemented!:")
            raise NotImplementedError
