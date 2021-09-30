# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from Polynomial_Regression import polynomial_regression as PR

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
is_show = True  # 控制是否绘图


def real_function(x):
    return np.sin(2 * np.pi * x)


def generate_data(num: int, sigma: float = 0.2):
    x_array = np.linspace(0, 1, num)
    y_array = real_function(x_array)
    noise_array = np.random.normal(0, sigma, num)
    y_array += noise_array
    return x_array, y_array


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
            plt.scatter(x_array, y_array, marker='o', color='red', s=20, label='sigma=' + str(sigma))
            # sin(2*PI*x)的图象
            plt.plot(draw_x, real_function(draw_x), color='black', linewidth=1.0, linestyle='-',
                     label="real func")
            plt.legend(loc='best')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("generate data with sigma = " + str(sigma))
        plt.tight_layout()
        plt.savefig(fname="different sigma", dpi=1000)
        plt.show()


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
    plt.savefig(fname="1", dpi=1000)
    plt.show()


def draw_rmse_order_graph(training_times: int = 1000, train_method="analytic", train_param=None):
    assert training_times > 0
    if is_show:
        mean_train_rmse_list = []
        mean_test_rmse_list = []
        for order in range(1, 10):
            mean_train_rmse = 0
            mean_test_rmse = 0
            for _ in range(training_times):
                pr = PR.Polynomial_Regression_Class(m=order, n_train=10, n_test=990, l2_norm_coefficient=0.,
                                                    verbose=False)
                w, train_rmse, test_rmse = None, None, None
                if train_method == "analytic":
                    w, train_rmse, test_rmse = pr.train(train_method, train_param, draw_result=False)
                elif train_method == "gradient descent":
                    w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method, train_param,
                                                                                     draw_result=True)
                mean_train_rmse += train_rmse
                mean_test_rmse += test_rmse
            mean_train_rmse_list.append(mean_train_rmse / training_times)
            mean_test_rmse_list.append(mean_test_rmse / training_times)
        draw_x = np.arange(1, 10)
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
        plt.savefig(fname="average rmse-order graph(" + train_method + ")", dpi=1000)
        plt.show()


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
        plt.title("average rmse_ln_l2_coefficient graph, training " + str(
            training_times) + " times per each coefficient(" + train_method + ")")
        plt.savefig(fname="rmse_ln_l2_coefficient", dpi=1000)
        plt.show()


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


def show_compare_regular():
    if is_show:
        fixed_train_data = generate_data(10, 0.2)
        fixed_test_data = generate_data(990, 0.2)
        train_rmse_list_without_regular, test_rmse_list_without_regular = [], []
        train_rmse_list_with_regular, test_rmse_list_with_regular = [], []
        order_range = [1, 3, 5, 7, 9]
        for order in order_range:
            # 不加正则
            pr_without_regular = PR.Polynomial_Regression_Class(m=order, n_train=10, n_test=990,
                                                                l2_norm_coefficient=0., verbose=False,
                                                                train_data=fixed_train_data, test_data=fixed_test_data)
            w_without_regular, train_rmse_without_regular, test_rmse_without_regular = pr_without_regular.train(
                draw_result=False)
            train_rmse_list_without_regular.append(train_rmse_without_regular)
            test_rmse_list_without_regular.append(test_rmse_without_regular)
            # 加正则
            pr_with_regular = PR.Polynomial_Regression_Class(m=order, n_train=10, n_test=990,
                                                             l2_norm_coefficient=np.exp(-9), verbose=False,
                                                             train_data=fixed_train_data, test_data=fixed_test_data)
            w_with_regular, train_rmse_with_regular, test_rmse_with_regular = pr_with_regular.train(draw_result=False)
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
        plt.savefig(fname="compare_regular", dpi=1000)
        plt.show()


def find_best_lr():
    lr_range = [0.33, 0.35, 0.37, 0.39]
    max_iters = 50000
    fixed_train_data = generate_data(10, 0.2)
    fixed_test_data = generate_data(990, 0.2)
    draw_color_list = ["red", "blue", "green", "black", "yellow", "purple", "pink"]
    for i in range(len(lr_range)):
        lr = lr_range[i]
        draw_color = draw_color_list[i]
        epsilon = 0
        pr = PR.Polynomial_Regression_Class(m=9, n_train=10, n_test=990, l2_norm_coefficient=np.exp(-9), verbose=False,
                                            train_data=fixed_train_data, test_data=fixed_test_data)
        w, train_rmse, test_rmse, iter_times, train_loss_list = pr.train(train_method="gradient descent",
                                                                         train_param=[lr, max_iters, epsilon])
        assert iter_times == max_iters
        assert len(train_loss_list) == iter_times + 1
        plt.plot(range(1000, iter_times + 1), np.array(train_loss_list)[1000:], color=draw_color, linewidth=1.0,
                 linestyle='-', label="lr=" + str(lr))
    plt.xlabel("iter_times")
    plt.ylabel("train_loss")
    plt.legend(loc="best")
    plt.title("loss-iter graph")
    plt.savefig(fname="compare_lr", dpi=1000)
    plt.show()


def show_compare_method():
    if is_show:
        m = 4
        fixed_train_data = generate_data(10, 0.2)
        fixed_test_data = generate_data(990, 0.2)
        pr = PR.Polynomial_Regression_Class(m=m, n_train=10, n_test=990, l2_norm_coefficient=np.exp(-9),
                                            verbose=False, train_data=fixed_train_data, test_data=fixed_test_data)
        analytic_w, analytic_train_rmse, analytic_test_rmse = pr.train("analytic")
        gd_w, gd_train_rmse, gd_test_rmse, iter_times, gd_train_loss_list = pr.train("gradient descent",
                                                                                     [0.39, 50000, 1e-8])

        cgd_w, cgd_train_rmse, cgd_test_rmse, iter_times, cgd_train_loss_list = pr.train("conjugate gradient descent",
                                                                                         [10, 1e-3])

        # plt.plot(range(0, iter_times + 1), np.array(cgd_train_loss_list), color="green", linewidth=1.0,
        #          linestyle='-')
        # plt.show()

        draw_x = np.linspace(0, 1, 200)
        analytic_predict_y = np.matmul(np.vander(draw_x, m + 1), analytic_w)
        gd_predict_y = np.matmul(np.vander(draw_x, m + 1), gd_w)
        cgd_predict_y = np.matmul(np.vander(draw_x, m + 1), cgd_w)
        plt.scatter(fixed_train_data[0], fixed_train_data[1], marker='o', color='green', s=10, label="train data")
        # 解析法得到的多项式函数图象
        plt.plot(draw_x, analytic_predict_y, color='blue', linewidth=1.0, linestyle='-',
                 label="predict func(analytic)")
        # 梯度下降法得到的多项式函数图象
        plt.plot(draw_x, gd_predict_y, color='red', linewidth=1.0, linestyle='-',
                 label="predict func(gradient descent)")
        # 共轭梯度下降法得到的多项式函数图象
        plt.plot(draw_x, cgd_predict_y, color='yellow', linewidth=1.0, linestyle='-',
                 label="predict func(conjugate gradient descent)")
        # sin(2*PI*x)的图象
        plt.plot(draw_x, real_function(draw_x), color='black', linewidth=1.0, linestyle='--',
                 label="real func")
        plt.legend(loc='best')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("predict func of different method")
        plt.savefig(fname="predict func of different method", dpi=1000)
        plt.show()


if __name__ == '__main__':
    # draw_data_generate()
    draw_rmse_order_graph()
    # draw_rmse_order_graph(10, "gradient descent", [0.1, 100000, 1e-6])
    # draw_rmse_l2_coefficient_graph()
    # find_best_l2_coefficient_graph()
    # show_compare_regular()
    # find_best_lr()
    # show_compare_method()
