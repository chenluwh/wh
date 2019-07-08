# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd


class Outliers:

    def lof(data, dimension=[], predict=None, k=5, epsilon=1.0, plot=False):
        # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
        try:
            if predict is None:
                predict = data.copy()
        except Exception:
            pass
        predict = pd.DataFrame(predict)
        if len(dimension) == 0:
            pass
        else:
            predict = predict.iloc[:, dimension]
        # 计算k距离和离群因子
        clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
        clf.fit(predict)
        # 记录 k 邻域距离
        predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
        # 记录 LOF 离群因子，做相反数处理
        predict['local outlier factor'] = - clf._decision_function(predict.iloc[:, :-1])
        outliers = predict[predict['local outlier factor'] > epsilon].iloc[:, 0:-2]
        inliers = predict[predict['local outlier factor'] <= epsilon].iloc[:, 0:-2]
        if np.shape(predict.iloc[:, 0:-2])[1] == 2:
            if plot is True:
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                plt.scatter(outliers[0], outliers[1], c='r', s=50, marker='.', alpha=None, label='离群点')
                plt.scatter(inliers[0], inliers[1], c='b', s=50, marker='.', alpha=None, label='正常点')
                plt.title('LOF局部离群点检测', fontsize=13)
                plt.legend()
                plt.show()

        return outliers, inliers

    def box(data, title):
        predict = data.copy()
        predict = pd.DataFrame(predict)
        # print(predict)
        # print(predict[0])
        if np.shape(predict)[1] == 1:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            p = plt.boxplot(predict[0])  # 绘制箱线图
            plt.title(title, fontsize=13)  # 设置箱线图的名称
            plt.show()
            outliers = p['fliers'][0].get_ydata()  # 获取异常值
            inliers = np.zeros(len(predict[0]) - len(outliers))
            # 获取异常数据索引
            outliers_index = []
            for j in range(0, len(outliers)):
                p = predict[predict[0] == outliers[j]].index.tolist()
                outliers_index.append(p)
            # 获取正常数据集合
            i = 0
            for e in predict[0]:
                if e not in outliers:
                    inliers[i] = e
                    i += 1
            return inliers, outliers, outliers_index
        else:
            print("输入数据集应该为一维")
