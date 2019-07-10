# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd


class Outliers:

    def lof(data, dimension=[], k=5, epsilon=1.0, plot=False):
        predict = data
        predict = pd.DataFrame(predict)
        # 检查是否传入维度参数
        if len(dimension) == 0:
            pass
        else:
            predict = predict.iloc[:, dimension]
        # 检查用于检测的数据集是否为二维以上数据集
        if np.shape(predict)[1] < 2:
            print("检测数据集必须为二维以上数据集")
        else:
            # 计算离群因子
            clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
            # 使用predict作为训练数据拟合模型
            clf.fit(predict)
            # 记录 LOF 离群因子，做相反数处理
            factor = - clf.negative_outlier_factor_  # 访问negative_outlier_factor_属性返回训练样本的异常分数的相反数
            factor = pd.DataFrame(factor)
            _outliers_index = factor[factor[0] > epsilon].index.tolist()
            _inliers_index = factor[factor[0] <= epsilon].index.tolist()
            _outliers = predict.iloc[_outliers_index, :]
            _inliers = predict.iloc[_inliers_index, :]
            # 如果检测数据集为二维，且plot=True，则绘制散点图
            if np.shape(predict)[1] == 2:
                if plot is True:
                    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    # 绘制离群点散点图
                    plt.scatter(_outliers.iloc[:, 0], _outliers.iloc[:, 1], c='r', s=50, marker='.',
                                alpha=None, label='离群点')
                    # 绘制正常点散点图
                    plt.scatter(_inliers.iloc[:, 0], _inliers.iloc[:, 1], c='b', s=50, marker='.',
                                alpha=None, label='正常点')
                    plt.title('LOF局部离群点检测', fontsize=13)
                    plt.legend()
                    plt.show()
            _outliers = data[_outliers_index, :]
            _inliers = data[_inliers_index, :]
            return _inliers, _inliers_index, _outliers, _outliers_index

    def box(data, title):
        predict = data
        predict = pd.DataFrame(predict)
        if not(np.shape(predict)[1] == 1):
            print("输入数据集应该为一维")
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            box = plt.boxplot(predict[0])  # 绘制箱线图
            plt.title(title, fontsize=13)  # 设置箱线图的名称
            plt.show()
            _outliers = box['fliers'][0].get_ydata()  # 获取异常值
            # 获取异常数据索引
            _outliers_index = []
            for j in range(0, len(_outliers)):
                index = predict[predict[0] == _outliers[j]].index.tolist()  # 获取索引并转成list类型
                _outliers_index.append(index[0])
            # 获取正常值索引
            _inliers_index = []
            for i in range(np.shape(predict)[0]):
                if i not in _outliers_index:
                    _inliers_index.append(i)
            # 获取正常数据集
            _inliers = data[_inliers_index, :]
            # 获取异常数据集
            _outliers = data[_outliers_index, :]
            return _inliers, _inliers_index, _outliers, _outliers_index
