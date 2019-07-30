# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans


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
            box = plt.boxplot(x=predict.values,  # 传入的数据
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
                index = predict[predict.values == _outliers[j]].index.tolist()  # 获取索引并转成list类型
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

    def dbscan(data, eps=None, minpts=None, dimension=[], plot=False):
        predict = data.copy()
        predict = pd.DataFrame(predict)
        # 检查是否传入维度参数
        if len(dimension) == 0:
            pass
        else:
            predict = predict.iloc[:, dimension]
        # 检查是否传入参数，若没有传入参数，则自适应确定参数
        X = predict.values
        tree = KDTree(X)
        if eps is not None and minpts is not None:
            optimal_eps = eps
            optimal_minpts = minpts
        else:
            optimal_minpts = 5
            dist, ind = tree.query(X, 5)
            mean = np.mean(dist[:, 4])
            std = np.std(dist[:, 4])
            optimal_eps = mean + std
        # 获取每个数据点在确定的eps邻域内包含的数据点数
        point_num = tree.query_radius(X, optimal_eps, count_only=True)
        print("eps的取值：", optimal_eps)
        print("minpts的取值", optimal_minpts)
        # 检查用于检测的数据集是否为二维以上数据集
        if np.shape(predict)[1] < 2:
            print("检测数据集必须为二维以上数据集")
        else:
            clu = DBSCAN(optimal_eps, optimal_minpts)
            clu.fit(predict)
            core_inliers_index = set(clu.core_sample_indices_.tolist())  # 属性core_sample_indices_返回核心点索引
            labels = clu.labels_  # 获取簇的标签，属于标签-1的为异常点
            predict['labels'] = labels
            unique_labels = list(set(labels))
            if -1 in unique_labels:
                unique_labels.remove(-1)
            else:
                pass
            # 获取正常点索引
            _inliers_index = predict[predict['labels'] != -1].index.tolist()
            # 获取离群值的索引
            _outliers_index = predict[predict['labels'] == -1].index.tolist()
            _outliers = predict.iloc[_outliers_index, :-1]
            _inliers = predict.iloc[_inliers_index, :-1]
            n = len(unique_labels)  # 获取正常点簇的个数
            if type(data) == list:
                data_1 = np.array(data)
                _outliers = data_1[_outliers_index, :].tolist()
                _inliers = data_1[_inliers_index, :].tolist()
            else:
                pass
            if type(data) == np.ndarray:
                _outliers = data[_outliers_index, :]
                _inliers = data[_inliers_index, :]
            else:
                pass
            if type(data) == pd.DataFrame:
                _outliers = data.iloc[_outliers_index, :]
                _inliers = data.iloc[_inliers_index, :]
            else:
                pass
            # 检查输入数据集是否为二维
            if np.shape(predict)[1] - 1 == 2:  # 由于加了一列存储分类标签，所以需要-1
                # 检查传入参数plot,是否有绘图需要
                if plot is True:
                    # 标红离群点
                    plt.plot(predict.iloc[_outliers_index, 0],
                             predict.iloc[_outliers_index, 1],
                             'o',
                             markerfacecolor='r',
                             markeredgecolor='r', markersize=3)
                    # 给不同簇中的数据点分配不同的颜色
                    # 定义一个list用来保存colors.py文件内对不同颜色设置的字符
                    if n <= 10:
                        colors = [(0, 1.0, 0, 1.0), (1.0, 0.843137254901961, 0, 1.0), (0, 0.501960784313725, 0, 1.0),
                                  (0.501960784313725, 0, 0.501960784313725, 1.0), (1.0, 1.0, 0, 1.0), (0, 0, 1.0, 1.0),
                                  (0, 1.0, 1.0, 1.0), (0.529411764705882, 0.807843137254902, 0.92156862745098, 1.0),
                                  (0.933333333333333, 0.509803921568627, 0.933333333333333, 1.0),
                                  (0, 0, 0, 1.0)]
                    else:
                        colors = [(0, 1.0, 0, 1.0), (1.0, 0.843137254901961, 0, 1.0), (0, 0.501960784313725, 0, 1.0),
                                  (0.501960784313725, 0, 0.501960784313725, 1.0), (1.0, 1.0, 0, 1.0), (0, 0, 1.0, 1.0),
                                  (0, 1.0, 1.0, 1.0), (0.529411764705882, 0.807843137254902, 0.92156862745098, 1.0),
                                  (0.933333333333333, 0.509803921568627, 0.933333333333333, 1.0),
                                  (0, 0, 0, 1.0)]
                        num = n - 10
                        colors1 = [plt.cm.Spectral(each) for each in np.linspace(0, 1, num)]
                        colors[len(colors):len(colors)] = colors1
                    for k, col in zip(unique_labels, colors):
                        # 区分核心点与非核心点的大小
                        plot_index = predict[predict['labels'] == k].index.tolist()
                        color = tuple(col)
                        core_plot = []
                        notcore_plot = []
                        for j in plot_index:
                            # 同簇数据点颜色相同，同簇内核心点较大，非核心点较小
                            if j in core_inliers_index:
                                core_plot.append(j)
                            else:
                                notcore_plot.append(j)
                        plt.plot(predict.iloc[core_plot, 0],
                                 predict.iloc[core_plot, 1],
                                 'o',
                                 markerfacecolor=color,
                                 markeredgecolor=color, markersize=3)
                        plt.plot(predict.iloc[notcore_plot, 0],
                                 predict.iloc[notcore_plot, 1],
                                 'o',
                                 markerfacecolor=color,
                                 markeredgecolor=color, markersize=1)
                    plt.show()
                else:
                    pass
            else:
                pass
            refer = data.copy()
            refer = pd.DataFrame(refer)
            # 将数据点在eps邻域内包含的数据点个数附在数据后面作为参考
            refer['eps邻域内点数'] = point_num
            return _inliers, _inliers_index, _outliers, refer

    def kmeans(data, dimension=[], k=None, k_max=None, minpoint=None, plot=False):
        predict = data
        predict = pd.DataFrame(predict)
        # 检查是否传入维度参数
        if len(dimension) == 0:
            pass
        else:
            predict = predict.iloc[:, dimension]
        data_num = len(predict)
        # 检查用于检测的数据集是否为二维以上数据集
        if np.shape(predict)[1] < 2:
            print("检测数据集必须为二维以上数据集")
        else:
            # 检查是否传入k值
            if k is None:
                # 检查是否传入搜寻最佳k值的上限
                if k_max is None:
                    k_max = 500
                else:
                    pass
                X = predict.values
                # 手肘法确定最佳k
                SSE = np.zeros(k_max)  # 存放每次结果的误差平方和
                for i in range(2, k_max):
                    estimator = KMeans(n_clusters=i)
                    estimator.fit(X)
                    SSE[i - 2] = estimator.inertia_  # 样本到其最近聚类中心的平方距离之和
                    # 搜寻肘弯处，搜寻到则跳出循环
                    if i > 2 and SSE[i - 2] / SSE[i - 3] > 0.99:
                        k = i
                        break
                    else:
                        pass
            else:
                pass
            print("k的取值：", k)
            clu = KMeans(n_clusters=k)
            clu.fit(predict)
            # 获取分类标签
            labels = clu.labels_
            predict['labels'] = labels
            # 获取不同簇的数据
            clus = []
            _index = [[] for i in range(0, k)]
            for i in range(0, k):
                index = predict[predict['labels'] == i].index.tolist()
                _index[i] = index
                clus.append(predict.iloc[index, : -1])
            # 检查是否传入判定簇是否为离群簇的点数下限
            if minpoint is None:
                minpoint = data_num // (2 * k)
            else:
                pass
            outliers_clus_index = []
            # 根据簇内所含数据点数判断该簇是否为离群点簇
            for i in range(0, k):
                # 如果簇内点数小于minpoint，则该簇为离群点簇
                if len(clus[i]) < minpoint:
                    outliers_clus_index.append(i)
                # 若簇内点数大于2，则计算簇心处的局部密度
                else:
                    pass
            inliers_index = []
            outliers_index = []
            for i in range(0, k):
                if i in outliers_clus_index:
                    for m in _index[i]:
                        outliers_index.append(m)
                else:
                    for j in _index[i]:
                        inliers_index.append(j)
            inliers_index = sorted(inliers_index)
            # 检查输入数据集是否为二维
            if np.shape(predict)[1] - 1 == 2:  # 由于加了一列存储分类标签，所以需要-1
                # 检查传入参数plot,是否有绘图需要
                if plot is True:
                    # 给不同簇中的数据点分配不同的颜色
                    if k <= 10:
                        colors = [(0, 1.0, 0, 1.0), (1.0, 0.843137254901961, 0, 1.0), (0, 0.501960784313725, 0, 1.0),
                                  (0.501960784313725, 0, 0.501960784313725, 1.0), (1.0, 1.0, 0, 1.0), (0, 0, 1.0, 1.0),
                                  (0, 1.0, 1.0, 1.0), (0.529411764705882, 0.807843137254902, 0.92156862745098, 1.0),
                                  (0.933333333333333, 0.509803921568627, 0.933333333333333, 1.0),
                                  (0, 0, 0, 1.0)]
                    else:
                        # 定义一个list用来保存colors.py文件内对不同颜色设置的字符
                        colors = [(0, 1.0, 0, 1.0), (1.0, 0.843137254901961, 0, 1.0), (0, 0.501960784313725, 0, 1.0),
                                  (0.501960784313725, 0, 0.501960784313725, 1.0), (1.0, 1.0, 0, 1.0), (0, 0, 1.0, 1.0),
                                  (0, 1.0, 1.0, 1.0), (0.529411764705882, 0.807843137254902, 0.92156862745098, 1.0),
                                  (0.933333333333333, 0.509803921568627, 0.933333333333333, 1.0),
                                  (0, 0, 0, 1.0)]
                        num = k - 10
                        colors1 = [plt.cm.Spectral(each) for each in np.linspace(0, 1, num)]
                        colors[len(colors):len(colors)] = colors1
                    # 定义一个list用来保存colors.py文件内对不同颜色设置的字符
                    for k, col in zip(range(0, k), colors):
                        if k in outliers_clus_index:
                            # 红色用于离群点上
                            plt.plot(predict[predict['labels'] == k].iloc[:, 0],
                                     predict[predict['labels'] == k].iloc[:, 1],
                                     'o',
                                     markerfacecolor='r',
                                     markeredgecolor='r', markersize=3)
                        else:
                            plt.plot(predict[predict['labels'] == k].iloc[:, 0],
                                     predict[predict['labels'] == k].iloc[:, 1],
                                     'o',
                                     markerfacecolor=tuple(col),
                                     markeredgecolor=tuple(col), markersize=3)
                else:
                    pass
                plt.show()
            else:
                pass
            _inliers = predict.iloc[inliers_index, : -1]
            _outliers = predict.iloc[outliers_index, : -1]
            if type(data) == list:
                data_1 = np.array(data)
                _inliers = data_1[inliers_index, :].tolist()
                _outliers = data_1[outliers_index, :].tolist()
            else:
                pass
            if type(data) == np.ndarray:
                _inliers = data[inliers_index, :]
                _outliers = data[outliers_index]
            else:
                pass
            if type(data) == pd.DataFrame:
                pass
            return _inliers, inliers_index, _outliers
