# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN


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
            clf = LocalOutlierFactor(n_neighbors=k, algorithm='auto', contamination=0.1, n_jobs=-1)
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
            if type(data) == list:
                data_1 = np.array(data)
                _outliers = data_1[_outliers_index, :]
                _inliers = data_1[_inliers_index, :]
                _outliers = _outliers.tolist()
                _inliers = _inliers.tolist()
            if type(data) == np.ndarray:
                _outliers = data[_outliers_index, :]
                _inliers = data[_inliers_index, :]
            if type(data) == pd.DataFrame:
                _outliers = data.iloc[_outliers_index, :]
                _inliers = data.iloc[_inliers_index, :]
            return _inliers, _inliers_index, _outliers, factor

    def box(data, whis=None, plot=True):
        predict = data
        predict = pd.DataFrame(predict)
        if not(np.shape(predict)[1] == 1):
            print("输入数据集应该为一维")
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            box = plt.boxplot(x=predict[0],  # 传入的数据
                              notch=False,  # 是否产生一个缺口盒子图，否则生成矩形箱图
                              vert=True,  # 是否垂直
                              whis=whis,  # 确定胡须超出第一和第三四分位数的距离，距离为whis*IQR
                              )  # 绘制箱线图
            # 检查传入plot参数，看是否有显示箱线图的需求
            if plot is True:
                plt.show()
            else:
                pass
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
            _inliers = predict.iloc[_inliers_index, :]  # 获取正常值
            # 获取正常、异常数据集
            if type(data) == list:
                data_1 = np.array(data)
                _inliers = data_1[_inliers_index, :]
                _outliers = data_1[_outliers_index, :]
                _outliers = _outliers.tolist()
                _inliers = _inliers.tolist()
            elif type(data) == np.ndarray:
                _outliers = data[_outliers_index, :]
                _inliers = data[_inliers_index, :]
            elif type(data) == pd.DataFrame:
                _outliers = data.iloc[_outliers_index]
                _inliers = data.iloc[_inliers_index]
            return _inliers, _inliers_index, _outliers

    def dbscan(data, dimension=[], eps=0.5, min_samples=5, plot=False):
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
            clu = DBSCAN(eps, min_samples)
            clu.fit(predict)
            core_inliers_index = clu.core_sample_indices_  # 属性core_sample_indices_返回核心点索引
            labels = clu.labels_  # 获取簇的标签，属于标签-1的为异常点
            predict['labels'] = labels
            _inliers_index = predict[predict['labels'] != -1].index.tolist()
            unique_labels = list(set(labels))
            # 获取不同簇的数据
            n = len(unique_labels)  # 获取簇的个数,所有离群点为一簇
            # 检查输入数据集是否为二维
            if np.shape(predict)[1] - 1 == 2:  # 由于加了一列存储分类标签，所以需要-1
                # 检查传入参数plot,是否有绘图需要
                if plot is True:
                    # 给不同簇中的数据点分配不同的颜色
                    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n)]
                    # 定义一个list用来保存colors.py文件内对不同颜色设置的字符
                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            # 红色用于离群点上
                            plt.plot(predict[predict['labels'] == k].iloc[:, 0],
                                     predict[predict['labels'] == k].iloc[:, 1],
                                     'o',
                                     markerfacecolor='r',
                                     markeredgecolor='r', markersize=3)
                        else:
                            predict_plot = predict[predict['labels'] == k]
                            core_plot = []
                            notcore_plot = []
                            for j in predict_plot.index.tolist():
                                # 将簇中的核心点和非核心点分开，核心点大，非核心点小，但同一簇中点颜色相同
                                if j in core_inliers_index:
                                    core_plot.append(j)
                                else:
                                    notcore_plot.append(j)
                                plt.plot(predict_plot.loc[core_plot, 0],
                                         predict_plot.loc[core_plot, 1],
                                         'o',
                                         markerfacecolor=tuple(col),
                                         markeredgecolor=tuple(col), markersize=6)
                                plt.plot(predict_plot.loc[notcore_plot, 0],
                                         predict_plot.loc[notcore_plot, 1],
                                         'o',
                                         markerfacecolor=tuple(col),
                                         markeredgecolor=tuple(col), markersize=3)
                else:
                    pass
                plt.show()
            else:
                pass
            # 获取离群值的索引
            _outliers_index = predict[predict['labels'] == -1].index.tolist()
            _outliers = predict.iloc[_outliers_index, :]
            _inliers = predict.iloc[_inliers_index, :]
            if type(data) == list:
                data_1 = np.array(data)
                _outliers = data_1[_outliers_index, :]
                _inliers = data_1[_inliers_index, :]
                _outliers = _outliers.tolist()
                _inliers = _inliers.tolist()
            if type(data) == np.ndarray:
                _outliers = data[_outliers_index, :]
                _inliers = data[_inliers_index, :]
            if type(data) == pd.DataFrame:
                data = data.iloc[:, :-1]
                _outliers = data.iloc[_outliers_index, :]
                _inliers = data.iloc[_inliers_index, :]
            return _inliers, _inliers_index, _outliers
