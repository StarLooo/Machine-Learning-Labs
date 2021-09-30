# -*- coding: UTF-8 -*-
import time

import numpy as np


class Decision_Tree:
    # 构造函数
    def __init__(self,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 features_name: np.ndarray,
                 n_train: int, n_test: int, n_features: int,
                 verbose: bool = True):
        """
        :param X_train: 训练集样本数据，shape = (n_train,n_features)
        :param y_train: 训练集样本对应的标签，shape = (n_train, )
        :param X_test: 测试集样本数据，shape = (n_test,n_features)
        :param y_test: 测试集样本对应的标签，shape = (n_test, )
        :param features_name: 样本各属性的名字，shape = (n_features, )
        :param n_features: 属性的个数
        :param verbose: 是否输出额外的提示信息
        :return 参数对应的初始化Decision_Tree对象
        """
        assert len(features_name) == n_features
        assert X_train.shape == (n_train, n_features) and y_train.shape == (n_train,)
        assert X_test.shape == (n_test, n_features) and y_test.shape == (n_test,)
        self.features_name = features_name
        self.features_name_to_index = {}
        for index in range(n_features):
            self.features_name_to_index[self.features_name[index]] = index
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.n_train, self.n_test, self.n_features = n_train, n_test, n_features
        self.verbose = verbose
        self.tree_dict = None

    # 判断当前训练集中剩余样本在剩余未划分属性上的取值是否一致
    def __is_same_samples(self, rest_samples: np.ndarray, rest_features_indexes: set):
        """
        :param rest_samples: 当前训练集中还未被划分到叶节点的样本
        :param rest_features_indexes: 当前训练集中还未被划分完成的属性对应的下标
        :return: 当前训练集中剩余样本在剩余未划分属性上的取值一致，则返回True；否则返回False
        """
        rest_samples_num = len(rest_samples)
        assert rest_samples_num >= 2  # 从逻辑上来说，当调用__is_same_samples()时保证rest_samples_num >= 2
        rest_features_indexes_list = list(rest_features_indexes)
        same_sample = rest_samples[0, rest_features_indexes_list]
        for row_index in range(1, rest_samples_num):
            if np.array((same_sample != rest_samples[row_index, rest_features_indexes_list])).any():
                return False
        return True

    # 找出当前训练集中剩余样本中所占比例最大的类别
    def __find_major_label(self, rest_labels: np.ndarray):
        """
        :param rest_labels: 当前训练集中各剩余样本的类别标签
        :return: 当前训练集中剩余样本中所占比例最大的类别
        """
        rest_samples_num = len(rest_labels)
        assert rest_samples_num >= 2  # 从逻辑上来说，当调用__find_major_label()时保证rest_samples_num >= 2
        labels_cnt_dict = {}
        for i in range(rest_samples_num):
            this_label = rest_labels[i]
            if this_label in labels_cnt_dict.keys():
                labels_cnt_dict[this_label] += 1
            else:
                labels_cnt_dict[this_label] = 1
        sorted_labels_cnt_list = sorted(labels_cnt_dict.items(), key=lambda x: x[1], reverse=True)
        major_label = sorted_labels_cnt_list[0][0]
        return major_label

    # 寻找最优划分属性
    def __search_best_divide_feature(self, rest_samples: np.ndarray, rest_labels: np.ndarray,
                                     rest_features_indexes: np.ndarray, gain_func: str = "information_gain"):
        """
        :param rest_samples: 当前训练集中还未被划分到叶节点的样本
        :param rest_labels: 当前训练集中各剩余样本的类别标签
        :param rest_features_indexes: 当前训练集中还未被划分完成的属性对应的下标
        :return: best_feature_index: 最佳划分属性的下标
        :return: best_gain: 对应最佳划分所能获得最大信息增益
        """
        assert len(rest_features_indexes) >= 1
        best_gain = -1
        best_feature_index = -1
        for rest_feature_index in rest_features_indexes:
            conditions = rest_samples[:, rest_feature_index]
            labels = rest_labels
            assert len(conditions) == len(labels)
            if gain_func == "information_gain":
                gain = self.__information_gain(conditions, labels)
            else:
                raise NotImplementedError()  # not implement
            assert 1 >= gain >= 0
            # 如果有多个可以获得最大增益对应的划分属性，选择遍历到的最后一个
            if gain >= best_gain:
                best_gain = gain
                best_feature_index = rest_feature_index
        assert best_feature_index >= 0
        return best_feature_index, best_gain

    # 递归生成决策树
    def __generate_tree(self, rest_samples_indexes: np.ndarray, rest_features_indexes: np.ndarray):
        """
        :param rest_samples_indexes: 当前训练集中还未被划分到叶节点的样本在训练集中所有样本中的下标
        :param rest_features_indexes: 当前训练集中还未被划分完成的属性对应的下标
        :return: 表示通过训练集训练出的决策树的字典
        """
        if self.verbose:
            print("--------------------------------------------------------------------")
            print('开始构造节点, 当前剩余待划分样本数:', len(rest_features_indexes), ',剩余可用于划分的属性数:',
                  len(rest_samples_indexes), "分别为:")
            print(features_name[list(rest_features_indexes)])
        rest_samples = self.X_train[list(rest_samples_indexes)]
        rest_labels = self.y_train[list(rest_samples_indexes)]
        rest_different_labels = {i for i in rest_labels}

        if len(rest_different_labels) == 1:
            node_label = rest_labels[0]
            if self.verbose:
                print("剩余的所有样本的类别标签一致，故该结点为叶节点，其类别标记为:", rest_labels[0])
            return node_label

        if len(rest_features_indexes) == 0 or self.__is_same_samples(rest_samples, rest_features_indexes):
            node_label = self.__find_major_label(rest_labels)
            if self.verbose:
                print("当前无可用于划分的属性或者剩余的所有样本在剩余属性域上取值相同，故该结点为叶节点，其类别标记为:")
            return node_label

        best_feature_index, best_gain = self.__search_best_divide_feature(rest_samples, rest_labels,
                                                                          rest_features_indexes)
        best_feature_name = self.features_name[best_feature_index]
        if self.verbose:
            print("选择的最佳划分属性:", best_feature_name, ",对应最大增益:", round(best_gain, 4))
        best_feature_values_set = set(sample[best_feature_index] for sample in self.X_train)
        tree_dict = {best_feature_name: {}}
        new_rest_features_indexes = rest_features_indexes.copy()  # 注意这里要copy,不然递归的时候内存别名会导致BUG
        new_rest_features_indexes.remove(best_feature_index)
        for best_feature_value in best_feature_values_set:
            if self.verbose:
                print("选择的最佳划分属性:", best_feature_name, "上的一个可能取值:", best_feature_value)
            new_rest_samples_indexes = set()
            for rest_sample_index in rest_samples_indexes:
                if self.X_train[rest_sample_index][best_feature_index] == best_feature_value:
                    new_rest_samples_indexes.add(rest_sample_index)

            if len(new_rest_samples_indexes) == 0:
                node_label = self.__find_major_label(rest_labels)
                if self.verbose:
                    print("剩余样本中在该属性上不能取到值:", best_feature_value, ",故该结点为叶节点，其类别标记为:",
                          node_label)
                tree_dict[best_feature_name][best_feature_value] = node_label

            else:
                if self.verbose:
                    print("该结点不是叶节点，以该属性的该取值进行划分, 开始递归生成决策树子树")
                tree_dict[best_feature_name][best_feature_value] = self.__generate_tree(new_rest_samples_indexes,
                                                                                        new_rest_features_indexes)
        return tree_dict

    # 预测样本的分类
    def predict(self, sample: np.ndarray, current_tree: dict):
        assert sample.shape == (self.n_features,)
        y_predict = []
        current_judge_feature_name = list(current_tree.keys())[0]
        current_judge_feature_index = self.features_name_to_index[current_judge_feature_name]
        # print(current_judge_feature_name)
        # print(current_judge_feature_index)
        this_feature_value = sample[current_judge_feature_index]
        # print(this_feature_value)
        branch = current_tree[current_judge_feature_name][this_feature_value]
        if isinstance(branch, dict):
            return self.predict(sample, branch)
        else:
            return branch

    # 训练函数
    def train(self):
        init_rest_samples_indexes = set(np.arange(self.n_train))
        init_rest_features_indexes = set(np.arange(self.n_features))
        self.tree_dict = self.__generate_tree(init_rest_samples_indexes, init_rest_features_indexes)
        if self.verbose:
            print("--------------------------------------------------------------------")
            print("决策树生成成功:")
            print(self.tree_dict)
            print("--------------------------------------------------------------------")
        y_predict = []
        for sample in self.X_test:
            label_predict = self.predict(sample, self.tree_dict)
            print(sample)
            print(label_predict)
            y_predict.append(label_predict)
        y_predict = np.array(y_predict)
        print("test accuracy:", self.__compute_accuracy(y_predict, self.y_test))

    # 计算准确度
    def __compute_accuracy(self, y_predict, y_test):
        assert len(y_predict) == self.n_test
        correct_num = 0
        for i in range(n_test):
            if y_predict[i] == y_test[i]:
                correct_num += 1
        accuracy = correct_num / self.n_test
        return accuracy

    # 计算信息熵
    @classmethod
    def entropy(cls, labels: np.ndarray):
        """
        :param labels: 类别标签集
        :return: 该类别集的信息熵
        """
        num_labels = len(labels)
        labels_cnt_dict = {}
        entropy = 0.
        for label in labels:
            if label not in labels_cnt_dict.keys():
                labels_cnt_dict[label] = 1
            else:
                labels_cnt_dict[label] += 1
        for label in labels_cnt_dict.keys():
            prob = labels_cnt_dict[label] / num_labels
            assert 1. >= prob > 0
            entropy += - prob * np.log2(prob)
        assert 1. >= entropy >= 0
        return entropy

    # 计算条件熵
    @classmethod
    def conditional_entropy(cls, conditions: np.ndarray, labels: np.ndarray):
        """
        :param conditions: 样本条件
        :param labels: 类别标签集
        :return: 该labels类别集在样本条件conditions下的条件熵
        """
        assert len(conditions) == len(labels)
        num_conditions = len(conditions)
        num_labels = len(labels)
        conditions_cnt_dict = {}
        conditional_entropy = 0
        for condition in conditions:
            if condition not in conditions_cnt_dict.keys():
                conditions_cnt_dict[condition] = 1
            else:
                conditions_cnt_dict[condition] += 1
        for condition in conditions_cnt_dict.keys():
            prob = conditions_cnt_dict[condition] / num_labels
            assert 1. >= prob > 0.
            conditional_labels = labels[np.where(conditions == condition)[0]]
            conditional_entropy += prob * Decision_Tree.entropy(conditional_labels)
        assert 1. >= conditional_entropy >= 0.
        return conditional_entropy

        # 计算信息增益

    @classmethod
    def __information_gain(cls, conditions: np.ndarray, labels: np.ndarray):
        """
        :param conditions: 样本条件
        :param labels: 类别标签集
        :return: 该labels类别集在样本条件conditions下信息增益
        """
        return Decision_Tree.entropy(labels) - Decision_Tree.conditional_entropy(conditions, labels)


if __name__ == "__main__":
    # 开始时间
    start = time.time()
    n_train, n_test, n_features = 10, 7, 6
    features_name = np.array(['色泽', '根蒂', '敲击', '纹理', '脐部', '触感'])
    X = np.array([
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'],
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑']
    ])
    y = np.array(['好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜',
                  '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜'])
    train_indexes = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
    test_indexes = [3, 4, 7, 8, 10, 11, 12]
    X_train, y_train = X[train_indexes], y[train_indexes]
    X_test, y_test = X[test_indexes], y[test_indexes]
    decision_tree = Decision_Tree(X_train, y_train, X_test, y_test, features_name, n_train, n_test, n_features)
    decision_tree.train()
