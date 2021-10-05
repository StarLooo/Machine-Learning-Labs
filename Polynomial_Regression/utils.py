# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import abc
from abc import ABC
from Polynomial_Regression import polynomial_regression as PR

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
is_show = True  # 控制是否绘图


# sin(2*Pi*x)
def real_function(x):
    return np.sin(2 * np.pi * x)


# 生成数据
def generate_data(num: int, sigma: float = 0.2, real_func=real_function):
    """
    生成带均值为0的高斯噪声的数据点。

    :param num: 所需生成数据点的个数。
    :param sigma: 高斯分布的标准差,大于0。
    :param real_func: 真实函数。
    :return: x_array, y_array,其中x_array为生成的num个数据点的x值组成的数组,y_array为生成的num个数据点的y值组成的数组。
    """
    x_array = np.linspace(0, 1, num)
    y_array = real_function(x_array)
    noise_array = np.random.normal(0, sigma, num)
    y_array += noise_array
    return x_array, y_array


# 绘制不同sigma下生成数据情况的示意图
def draw_data_generate():
    if is_show:
        plt.figure(figsize=(12, 8))
        sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        cnt = 0
        for sigma in sigma_list:
            cnt += 1
            draw_x = np.linspace(0, 1, 200)
            x_array, y_array = generate_data(num=20, sigma=sigma)
            plt.subplot(2, 3, cnt)
            # 训练集中样本散点图
            plt.scatter(x_array, y_array, marker='o', color='red', s=20, label='data')
            # sin(2*PI*x)的图象
            plt.plot(draw_x, real_function(draw_x), color='black', linewidth=1.0, linestyle='-',
                     label="real func")
            plt.legend(loc='best')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("generate data with sigma = " + str(sigma))
        plt.tight_layout()
        plt.savefig(fname="different sigma.svg", dpi=10000, format="svg")
        plt.show()


# 训练结束后对训练结果进行可视化
def draw_predict_analysis(train_x_array, train_y_array, draw_x, draw_predict_y, title):
    # 训练集中样本散点图
    plt.scatter(train_x_array, train_y_array, marker='o', color='blue', s=20, label='train data')
    # 训练得到的多项式函数图象
    plt.plot(draw_x, draw_predict_y, color='black', linewidth=1.0, linestyle='--',
             label="predict func(analytic)")
    # sin(2*PI*x)的图象
    plt.plot(draw_x, real_function(draw_x), color='green', linewidth=1.0, linestyle='-',
             label="real func")
    plt.legend(loc='best')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    print(title)
    plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
    plt.show()


# 绘制rmse随多项式阶数的变化图
def draw_rmse_order_graph(training_times: int = 1000, train_method="analytic", train_param=None):
    assert training_times > 0
    if is_show:
        mean_train_rmse_list = []
        mean_test_rmse_list = []
        order_range = range(1, 10)
        for order in order_range:
            mean_train_rmse = 0
            mean_test_rmse = 0
            for _ in range(training_times):
                pr = PR.Polynomial_Regression_Class(m=order, n_train=10, n_test=990, l2_norm_coefficient=0.,
                                                    verbose=True)
                w, train_rmse, test_rmse = None, None, None
                if train_method == "analytic":
                    w, train_rmse, test_rmse = pr.train(train_method, train_param, draw_result=False)
                elif train_method == "gradient descent":
                    w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method, train_param,
                                                                                     draw_result=False)
                elif train_method == "stochastic gradient descent":
                    w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method, train_param,
                                                                                     draw_result=False)
                elif train_method == "els gradient descent":
                    w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method, train_param,
                                                                                     draw_result=False)
                else:
                    raise NotImplementedError
                mean_train_rmse += train_rmse
                mean_test_rmse += test_rmse
            mean_train_rmse_list.append(mean_train_rmse / training_times)
            mean_test_rmse_list.append(mean_test_rmse / training_times)
        draw_x = order_range
        plt.scatter(draw_x, np.array(mean_train_rmse_list), marker='o', color='blue', s=15)
        plt.scatter(draw_x, np.array(mean_test_rmse_list), marker='o', color='red', s=15)
        plt.plot(draw_x, np.array(mean_train_rmse_list), color='blue', linewidth=1.0, linestyle='--',
                 label="train rmse")
        plt.plot(draw_x, np.array(mean_test_rmse_list), color='red', linewidth=1.0, linestyle='-',
                 label="test rmse")
        plt.xlabel("order")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.title(
            "rmse-order graph, training " + str(training_times) + " times per each order(" + train_method + ")")
        plt.savefig(fname="average rmse-order graph(" + train_method + ").svg", dpi=10000, format="svg")
        plt.show()


# 绘制不同样本点个数下的训练结果
def draw_different_samples():
    if is_show:
        n_train_range = [10, 25, 50, 75, 100]
        for n_train in n_train_range:
            mean_train_rmse = 0
            mean_test_rmse = 0
            pr = PR.Polynomial_Regression_Class(m=9, n_train=n_train, n_test=990, l2_norm_coefficient=0.,
                                                verbose=True)
            pr.train(draw_result=True)


# 绘制rmse随l2_coefficient的变化图
def draw_rmse_l2_coefficient_graph(training_times: int = 100, train_method="analytic", train_param=None):
    assert training_times > 0
    if is_show:
        mean_train_rmse_list = []
        mean_test_rmse_list = []
        ln_l2_coefficient_range = range(-50, 1)
        fixed_train_data = generate_data(10, 0.2)
        fixed_test_data = generate_data(990, 0.2)
        for ln_l2_coefficient in ln_l2_coefficient_range:
            mean_train_rmse = 0
            mean_test_rmse = 0
            for _ in range(training_times):
                pr = PR.Polynomial_Regression_Class(m=9, n_train=10, n_test=990,
                                                    l2_norm_coefficient=np.exp(ln_l2_coefficient), verbose=False,
                                                    train_data=fixed_train_data, test_data=fixed_test_data)
                w, train_rmse, test_rmse = None, None, None
                if train_method == "analytic":
                    w, train_rmse, test_rmse = pr.train(train_method, train_param)
                elif train_method == "gradient descent":
                    w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method, train_param)
                mean_train_rmse += train_rmse
                mean_test_rmse += test_rmse
            mean_train_rmse_list.append(mean_train_rmse / training_times)
            mean_test_rmse_list.append(mean_test_rmse / training_times)
        draw_x = ln_l2_coefficient_range
        plt.scatter(draw_x, np.array(mean_train_rmse_list), marker='o', color='blue', s=10)
        plt.scatter(draw_x, np.array(mean_test_rmse_list), marker='o', color='red', s=10)
        plt.plot(draw_x, np.array(mean_train_rmse_list), color='blue', linewidth=1.0, linestyle='--',
                 label="train rmse")
        plt.plot(draw_x, np.array(mean_test_rmse_list), color='red', linewidth=1.0, linestyle='-',
                 label="test rmse")
        plt.xlabel("ln(l2_coefficient)")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.title("rmse_ln_l2_coefficient graph(" + train_method + ")")
        plt.savefig(fname="rmse_ln_l2_coefficient.svg", dpi=10000, format="svg")
        plt.show()


# 多次实验然后投票确定最佳正则系数
def find_best_l2_coefficient_graph(training_times: int = 1000):
    assert training_times > 0

    ln_l2_coefficient_range = range(-12, -4)
    mean_train_rmse_list = []
    mean_test_rmse_list = []
    for ln_l2_coefficient in ln_l2_coefficient_range:
        mean_train_rmse = 0
        mean_test_rmse = 0
        for _ in range(training_times):
            pr = PR.Polynomial_Regression_Class(m=9, n_train=10, n_test=990,
                                                l2_norm_coefficient=np.exp(ln_l2_coefficient), verbose=False)

            w, train_rmse, test_rmse = pr.train()
            mean_train_rmse += train_rmse
            mean_test_rmse += test_rmse
        mean_train_rmse_list.append(mean_train_rmse / training_times)
        mean_test_rmse_list.append(mean_test_rmse / training_times)
    best_ln_l2_coefficient = ln_l2_coefficient_range[np.argmin(np.array(mean_test_rmse_list))]
    print("best_ln_l2_coefficient:", best_ln_l2_coefficient)


# 对比加正则项后与加正则项前的拟合效果
def show_compare_regular():
    if is_show:
        fixed_train_data = generate_data(10, 0.2)
        fixed_test_data = generate_data(990, 0.2)
        train_rmse_list_without_regular, test_rmse_list_without_regular = [], []
        train_rmse_list_with_regular, test_rmse_list_with_regular = [], []
        order_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for order in order_range:
            # 不加正则
            pr_without_regular = PR.Polynomial_Regression_Class(m=order, n_train=10, n_test=990,
                                                                l2_norm_coefficient=0., verbose=False,
                                                                train_data=fixed_train_data, test_data=fixed_test_data)
            w_without_regular, train_rmse_without_regular, test_rmse_without_regular = pr_without_regular.train(
                draw_result=True)
            train_rmse_list_without_regular.append(train_rmse_without_regular)
            test_rmse_list_without_regular.append(test_rmse_without_regular)
            # 加正则
            pr_with_regular = PR.Polynomial_Regression_Class(m=order, n_train=10, n_test=990,
                                                             l2_norm_coefficient=np.exp(-9), verbose=False,
                                                             train_data=fixed_train_data, test_data=fixed_test_data)
            w_with_regular, train_rmse_with_regular, test_rmse_with_regular = pr_with_regular.train(draw_result=True)
            train_rmse_list_with_regular.append(train_rmse_with_regular)
            test_rmse_list_with_regular.append(test_rmse_with_regular)

        draw_x = order_range
        plt.subplot(1, 2, 1)
        plt.scatter(draw_x, np.array(train_rmse_list_without_regular), marker='o', color='blue', s=10)
        plt.scatter(draw_x, np.array(test_rmse_list_without_regular), marker='o', color='red', s=10)
        plt.plot(draw_x, np.array(train_rmse_list_without_regular), color='blue', linewidth=1.0, linestyle='--',
                 label="train rmse")
        plt.plot(draw_x, np.array(test_rmse_list_without_regular), color='red', linewidth=1.0, linestyle='-',
                 label="test rmse")
        plt.xlabel("order")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.title("rmse-order graph, without regular")

        plt.subplot(1, 2, 2)
        plt.scatter(draw_x, np.array(train_rmse_list_with_regular), marker='o', color='blue', s=10)
        plt.scatter(draw_x, np.array(test_rmse_list_with_regular), marker='o', color='red', s=10)
        plt.plot(draw_x, np.array(train_rmse_list_with_regular), color='blue', linewidth=1.0, linestyle='--',
                 label="train rmse")
        plt.plot(draw_x, np.array(test_rmse_list_with_regular), color='red', linewidth=1.0, linestyle='-',
                 label="test rmse")
        plt.xlabel("order")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.title("rmse-order graph, with regular")

        plt.tight_layout()
        plt.savefig(fname="compare_regular.svg", dpi=10000, format="svg")
        plt.show()


# 寻找最优学习率
def find_best_lr(train_method="gradient descent"):
    # 第一次搜索
    # lr_range = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 1.0]
    # 第二次搜索
    lr_range = [0.65, 0.75, 0.85, 0.95]
    max_iters = 150000
    fixed_train_data = generate_data(10, 0.2)
    fixed_test_data = generate_data(990, 0.2)
    draw_color_list = ["red", "blue", "green", "black", "yellow", "purple", "pink"]
    for i in range(len(lr_range)):
        lr = lr_range[i]
        draw_color = draw_color_list[i]
        epsilon = 1e-3
        pr = PR.Polynomial_Regression_Class(m=9, n_train=10, n_test=990, l2_norm_coefficient=np.exp(-9), verbose=True,
                                            train_data=fixed_train_data, test_data=fixed_test_data)
        train_loss_list = []
        iter_times = 0
        train_param = [lr, max_iters, epsilon]
        if train_method == "gradient descent":
            w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method, train_param,
                                                                             draw_result=False)
        elif train_method == "stochastic gradient descent":
            w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method, train_param,
                                                                             draw_result=False)
        else:
            raise NotImplementedError
        print("train loss:", train_loss_list[-1])
        assert len(train_loss_list) == iter_times + 1
        plt.plot(range(10, iter_times + 1), np.array(train_loss_list)[10:], color=draw_color, linewidth=1.0,
                 linestyle='-', label="lr=" + str(lr))
    plt.xlabel("iter_times")
    plt.ylabel("train_loss")
    plt.legend(loc="best")
    plt.title("loss-iter graph(" + train_method + ")")
    plt.savefig(fname="compare_lr(" + train_method + ").svg", dpi=10000, format="svg")
    plt.show()


# 绘制迭代次数随训练样本个数的变化图
def draw_iter_times_n_train_graph(train_method="gradient descent"):
    m = 9
    n_train_range = range(10, 100, 5)

    l2_norm_coefficient = 0.0
    iter_times_list = []
    for i in range(len(n_train_range)):
        n_train = n_train_range[i]
        fixed_train_data = generate_data(n_train, 0.2)
        fixed_test_data = generate_data(990, 0.2)
        pr = PR.Polynomial_Regression_Class(m=m, n_train=n_train, n_test=990, l2_norm_coefficient=l2_norm_coefficient,
                                            verbose=True, train_data=fixed_train_data, test_data=fixed_test_data)
        if train_method == "gradient descent":
            gd_w, gd_train_rmse, gd_test_rmse, gd_iter_times, gd_train_loss_list = pr.train(train_method,
                                                                                            [0.4, 200000, 1e-2])
            print("gradient descent iter_times:", gd_iter_times)
            iter_times_list.append(gd_iter_times)

        elif train_method == "stochastic gradient descent":
            sgd_w, sgd_train_rmse, sgd_test_rmse, sgd_iter_times, sgd_train_loss_list = pr.train(
                "stochastic gradient descent", [0.2, 100000, 1e-6])
            print("stochastic gradient descent iter_times:", sgd_iter_times)
            iter_times_list.append(sgd_iter_times)

        elif train_method == "els gradient descent":
            els_gd_w, els_gd_train_rmse, els_gd_test_rmse, els_gd_iter_times, els_gd_train_loss_list = pr.train(
                "els gradient descent", [100000, 1e-3])
            print("els gradient descent iter_times:", els_gd_iter_times)
            iter_times_list.append(els_gd_iter_times)

        elif train_method == "conjugate gradient descent":
            cgd_w, cgd_train_rmse, cgd_test_rmse, cgd_iter_times, cgd_train_loss_list = pr.train(
                "conjugate gradient descent", [m + 1, 1e-4])
            print("conjugate gradient descent iter_times:", cgd_iter_times)
            iter_times_list.append(cgd_iter_times)
        else:
            raise NotImplementedError

    assert len(n_train_range) == len(iter_times_list)
    plt.ylabel("iter_times")
    plt.xlabel("n_train")
    plt.plot(n_train_range, np.array(iter_times_list), color="black", linewidth=1.0, linestyle='-')
    title = "iter_times-n_train graph(" + train_method + ")"
    plt.title(title)
    plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
    plt.show()


# 绘制迭代次数随多项式阶数的变化图
def draw_iter_times_m_graph(train_method="gradient descent"):
    m = 9
    m_range = range(1, 20)

    l2_norm_coefficient = 0.0
    iter_times_list = []
    for m in m_range:
        fixed_train_data = generate_data(20, 0.2)
        fixed_test_data = generate_data(990, 0.2)
        pr = PR.Polynomial_Regression_Class(m=m, n_train=20, n_test=990, l2_norm_coefficient=l2_norm_coefficient,
                                            verbose=True, train_data=fixed_train_data, test_data=fixed_test_data)
        if train_method == "gradient descent":
            gd_w, gd_train_rmse, gd_test_rmse, gd_iter_times, gd_train_loss_list = pr.train(train_method,
                                                                                            [0.4, 200000, 1e-3])
            print("gradient descent iter_times:", gd_iter_times)
            iter_times_list.append(gd_iter_times)

        elif train_method == "stochastic gradient descent":
            sgd_w, sgd_train_rmse, sgd_test_rmse, sgd_iter_times, sgd_train_loss_list = pr.train(
                "stochastic gradient descent", [0.2, 100000, 1e-6])
            print("stochastic gradient descent iter_times:", sgd_iter_times)
            iter_times_list.append(sgd_iter_times)

        elif train_method == "els gradient descent":
            els_gd_w, els_gd_train_rmse, els_gd_test_rmse, els_gd_iter_times, els_gd_train_loss_list = pr.train(
                "els gradient descent", [100000, 1e-3])
            print("els gradient descent iter_times:", els_gd_iter_times)
            iter_times_list.append(els_gd_iter_times)

        elif train_method == "conjugate gradient descent":
            cgd_w, cgd_train_rmse, cgd_test_rmse, cgd_iter_times, cgd_train_loss_list = pr.train(
                "conjugate gradient descent", [m + 1, 1e-4])
            print("conjugate gradient descent iter_times:", cgd_iter_times)
            iter_times_list.append(cgd_iter_times)

        else:
            raise NotImplementedError

    assert len(m_range) == len(iter_times_list)
    plt.ylabel("iter_times")
    plt.xlabel("m")
    plt.plot(m_range, np.array(iter_times_list), color="black", linewidth=1.0, linestyle='-')
    title = "iter_times-m graph(" + train_method + ")"
    plt.title(title)
    plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
    plt.show()


# 对比不同方法
def show_compare_method(m, train_methods, l2_norm_coefficient=0.0):
    if is_show:
        draw_x = np.linspace(0, 1, 200)
        fixed_train_data = generate_data(10, 0.2)
        fixed_test_data = generate_data(990, 0.2)
        pr = PR.Polynomial_Regression_Class(m=m, n_train=10, n_test=990, l2_norm_coefficient=l2_norm_coefficient,
                                            verbose=True, train_data=fixed_train_data, test_data=fixed_test_data)

        if "analytic" in train_methods:
            analytic_w, analytic_train_rmse, analytic_test_rmse = pr.train("analytic")
            analytic_predict_y = np.matmul(np.vander(draw_x, m + 1, increasing=True), analytic_w)
            # 解析法得到的多项式函数图象
            plt.plot(draw_x, analytic_predict_y, color='blue', linewidth=1.0, linestyle='-',
                     label="predict func(analytic)")

        if "gradient descent" in train_methods:
            gd_w, gd_train_rmse, gd_test_rmse, gd_iter_times, gd_train_loss_list = pr.train("gradient descent",
                                                                                            [1., 150000, 1e-3])
            print("gradient descent iter_times:", gd_iter_times)
            gd_predict_y = np.matmul(np.vander(draw_x, m + 1, increasing=True), gd_w)
            # 梯度下降法得到的多项式函数图象
            plt.plot(draw_x, gd_predict_y, color='red', linewidth=1.0, linestyle='-',
                     label="predict func(gradient descent)")

        if "stochastic gradient descent" in train_methods:
            sgd_w, sgd_train_rmse, sgd_test_rmse, sgd_iter_times, sgd_train_loss_list = pr.train(
                "stochastic gradient descent", [0.2, 100000, 1e-6])
            print("stochastic gradient descent iter_times:", sgd_iter_times)
            sgd_predict_y = np.matmul(np.vander(draw_x, m + 1, increasing=True), sgd_w)
            # 随机梯度下降法得到的多项式函数图象
            plt.plot(draw_x, sgd_predict_y, color='green', linewidth=1.0, linestyle='-',
                     label="predict func(stochastic gradient descent)")

        if "els gradient descent" in train_methods:
            els_gd_w, els_gd_train_rmse, els_gd_test_rmse, els_gd_iter_times, els_gd_train_loss_list = pr.train(
                "els gradient descent", [100000, 1e-3])
            print("els gradient descent iter_times:", els_gd_iter_times)
            els_gd_predict_y = np.matmul(np.vander(draw_x, m + 1, increasing=True), els_gd_w)
            # 精确线搜索梯度下降法得到的多项式函数图象
            plt.plot(draw_x, els_gd_predict_y, color='purple', linewidth=1.0, linestyle='-',
                     label="predict func(els gradient descent)")

        if "conjugate gradient descent" in train_methods:
            cgd_w, cgd_train_rmse, cgd_test_rmse, cgd_iter_times, cgd_train_loss_list = pr.train(
                "conjugate gradient descent", [m + 1, 1e-4])
            print("conjugate gradient descent iter_times:", cgd_iter_times)
            cgd_predict_y = np.matmul(np.vander(draw_x, m + 1, increasing=True), cgd_w)
            # 共轭梯度下降法得到的多项式函数图象
            plt.plot(draw_x, cgd_predict_y, color='yellow', linewidth=1.0, linestyle='-',
                     label="predict func(conjugate gradient descent)")

        else:
            raise NotImplementedError

        # 数据散点图
        plt.scatter(fixed_train_data[0], fixed_train_data[1], marker='o', color='green', s=10, label="train data")
        # sin(2*PI*x)的图象
        plt.plot(draw_x, real_function(draw_x), color='black', linewidth=1.0, linestyle='--',
                 label="real func")
        plt.legend(loc='best')
        plt.xlabel("x")
        plt.ylabel("y")
        title = "predict func of different method,m=" + str(m)
        if l2_norm_coefficient > 0:
            title += ",l2_coef=" + str(round(l2_norm_coefficient, 4))
        plt.title(title)
        plt.savefig(fname=title + ".svg", dpi=10000, format="svg")
        plt.show()


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
            # 若梯度在误差允许范围内接近0则结束训练，退出循环
            if np.linalg.norm(latest_grad, 2) < self.epsilon:
                if self.verbose:
                    print("gradient descent finished, actual iter times:", actual_iter_times)
                break
            new_param = self.train_param - self.lr * latest_grad  # 梯度下降
            train_loss = self.loss_func(new_param)  # 计算本次迭代后的训练误差
            # 若loss不再下降，则不更新参数，并减小学习率
            if train_loss >= pre_loss:
                self.lr *= 0.1  # 减小学习率
                train_loss = pre_loss
            else:
                # 否则更新参数
                self.train_param = new_param
            # 若迭代次数达到上限，训练结束
            if actual_iter_times == self.max_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list, latest_grad


class ELS_Gradient_Descent_Optimizer(Optimizer, ABC):
    def __init__(self, train_param, hyper_params, loss_func, grad_func, els_func, verbose=False):
        super().__init__(train_param, hyper_params, verbose)
        self.max_iter_times, self.epsilon = hyper_params
        assert self.max_iter_times > 0 and self.epsilon >= 0.
        self.loss_func = loss_func
        self.grad_func = grad_func
        self.els_func = els_func

    def train(self):
        if self.verbose:
            print("optimize with gradient descent.")
        train_loss = self.loss_func(self.train_param)
        train_loss_list = []
        latest_grad = None
        actual_iter_times = 0
        for iter_num in range(1, self.max_iter_times + 1):
            # 若梯度的2范数小于阈值，则训练结束，退出循环
            if latest_grad is not None and np.linalg.norm(latest_grad, 2) < self.epsilon:
                if self.verbose:
                    print("gradient descent finished, actual iter times:", actual_iter_times)
                break
            actual_iter_times = iter_num
            pre_loss = train_loss  # 上一次迭代的loss
            train_loss_list.append(train_loss)  # 记录train_loss
            pre_param = self.train_param  # 上一次迭代的w
            latest_grad = self.grad_func(self.train_param)  # 求梯度
            step_len = self.els_func(latest_grad)  # 确定最优步长
            self.train_param = self.train_param - step_len * latest_grad  # 梯度下降
            train_loss = self.loss_func(self.train_param)  # 计算本次迭代后的训练误差

            # 若迭代次数达到上限，训练结束
            if actual_iter_times == self.max_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list, latest_grad


class Stochastic_Gradient_Descent_Optimizer(Optimizer, ABC):
    def __init__(self, train_param, hyper_params, n_samples, loss_func, grad_func, verbose=False):
        super().__init__(train_param, hyper_params, verbose)
        assert n_samples > 0
        self.n_samples = n_samples
        self.lr, self.max_iter_times, self.epsilon = hyper_params
        assert self.lr > 0. and self.max_iter_times > 0 and self.epsilon >= 0.
        self.loss_func = loss_func
        self.grad_func = grad_func

    def train(self):
        if self.verbose:
            print("optimize with stochastic gradient descent.")
        train_loss = self.loss_func(self.train_param)
        train_loss_list = []
        actual_iter_times = 0

        for iter_num in range(1, self.max_iter_times + 1):
            actual_iter_times = iter_num
            pre_loss = train_loss  # 上一次迭代的loss
            train_loss_list.append(train_loss)  # 记录train_loss

            for i in range(self.n_samples):
                partial_grad = self.grad_func(self.train_param, i)  # 求部分梯度
                self.train_param = self.train_param - self.lr * partial_grad  # 随机梯度下降
            train_loss = self.loss_func(self.train_param)  # 计算n_samples次迭代后的训练误差

            # 若loss变化很小则减小lr
            if abs(pre_loss - train_loss) < self.epsilon:
                if self.verbose:
                    print("gradient descent finished, actual iter times:", actual_iter_times)
                break
            # 若迭代次数达到上限，训练结束
            if actual_iter_times == self.max_iter_times:
                if self.verbose:
                    print("iter too many times, terminate train!")
        train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list


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
        d = r
        delta_new = np.dot(r, r)
        delta_init = delta_new
        train_loss_init = self.loss_func(self.train_param)
        train_loss_list.append(train_loss_init)
        while actual_iter_times <= self.max_iter_times and delta_new > (self.epsilon ** 2) * delta_init:
            q = np.matmul(self.A, d)
            alpha = delta_new / np.dot(d, q)
            self.train_param += alpha * d
            # r = self.b - np.matmul(self.A, self.train_param)
            r = r - alpha * q
            delta_old = delta_new
            delta_new = np.dot(r, r)
            beta = delta_new / delta_old
            d = r + beta * d
            actual_iter_times += 1
            # 计算本次迭代后的训练误差
            train_loss = self.loss_func(self.train_param)
            train_loss_list.append(train_loss)
        if actual_iter_times == self.max_iter_times:
            if self.verbose:
                print("iter too many times, terminate train!")
        else:
            if self.verbose:
                print("conjugate gradient finished, actual iter times:", actual_iter_times)

        # r = self.b - np.matmul(self.A, self.train_param)
        # p = r
        # for iter_num in range(1, self.max_iter_times + 1):
        #     actual_iter_times = iter_num
        #     pre_loss = train_loss  # 上一次迭代的loss
        #     train_loss_list.append(train_loss)  # 记录train_loss
        #     pre_param = self.train_param  # 上一次迭代的w
        #     old_r_inner_product = np.matmul(r.transpose(), r)
        #     alpha = old_r_inner_product / np.matmul(np.matmul(p.transpose(), self.A), p)
        #     self.train_param = self.train_param + alpha * p
        #     r = r - alpha * np.matmul(self.A, p)
        #     new_r_inner_product = np.matmul(r.transpose(), r)
        #     beta = new_r_inner_product / old_r_inner_product
        #     p = r + beta * p
        #
        #     # 计算本次迭代后的训练误差
        #     train_loss = self.loss_func()
        #     assert train_loss < pre_loss
        #     # 残差r的L1-norm小于阈值，则训练结束，退出循环
        #     if np.max(np.abs(r)) < self.epsilon:
        #         if self.verbose:
        #             print("conjugate gradient finished, actual iter times:", actual_iter_times)
        #         break
        #     if iter_num == actual_iter_times:
        #         if self.verbose:
        #             print("iter too many times, terminate train!")
        # train_loss_list.append(train_loss)

        return self.train_param, actual_iter_times, train_loss_list


if __name__ == '__main__':
    # draw_data_generate()
    # draw_rmse_order_graph(1)
    # draw_rmse_order_graph()
    # draw_different_samples()
    # draw_rmse_order_graph(1, train_method="els gradient descent", train_param=[50000, 1e-6])
    # draw_rmse_l2_coefficient_graph()
    # find_best_l2_coefficient_graph()
    # show_compare_regular()
    # find_best_lr()
    # for m in [3, 5, 7, 9]:
    #     show_compare_method(m=m, train_methods=["analytic", "gradient descent"])
    #     show_compare_method(m=m, train_methods=["analytic", "gradient descent"], l2_norm_coefficient=np.exp(-9))
    # draw_rmse_order_graph(1, train_method="stochastic gradient descent", train_param=[0.3, 50000, 1e-6])
    # draw_rmse_order_graph(1, train_method="gradient descent", train_param=[0.1, 50000, 1e-6])
    # draw_rmse_order_graph(10, "gradient descent", [0.1, 100000, 1e-6])
    # draw_iter_times_n_train_graph("gradient descent")
    # draw_iter_times_m_graph("gradient descent")
    # draw_iter_times_n_train_graph("conjugate gradient descent")
    # draw_iter_times_m_graph("conjugate gradient descent")
    # for m in [3, 5, 7, 9]:
    #     show_compare_method(m=m, train_methods=["analytic", "gradient descent", "conjugate gradient descent"])
    #     show_compare_method(m=m, train_methods=["analytic", "gradient descent", "conjugate gradient descent"],
    #                         l2_norm_coefficient=np.exp(-9))
    #     show_compare_method(m=m, train_methods=["analytic", "gradient descent", "conjugate gradient descent",
    #                                             "stochastic gradient descent", "els gradient descent", ],
    #                         l2_norm_coefficient=np.exp(-9))
    # draw_iter_times_n_train_graph("stochastic gradient descent")
    # draw_iter_times_m_graph("stochastic gradient descent")
    draw_iter_times_n_train_graph("els gradient descent")
    draw_iter_times_m_graph("els gradient descent")
