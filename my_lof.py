# -*- coding: UTF-8 -*-

from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


class LOF:
    def __init__(self, data, k, epsilon=1.0):
        self.data = data
        self.k = k
        self.epsilon = epsilon
        self.N = self.data.shape[0]  # 获取data第一维度的长度

    def get_dist(self):
        # 计算欧式距离矩阵
        return cdist(self.data, self.data)  # 计算两组输入每队之间的距离，返回一个矩阵

    def _kdist(self, arr):
        # 计算k距离
        # print(arr)
        inds_sort = np.argsort(arr)  # 对arr按照升序排列，返回arr数组从小到大的索引
        neighbor_ind = inds_sort[1:self.k+1]  # 邻域内点索引，取欧氏距离从小到大排列的前k个距离
        # print(inds_sort)
        # print(neighbor_ind)
        # print(arr[neighbor_ind[-1]])
        return arr[neighbor_ind[-1]]  # 返回每个点的k距离

    def get_rdist(self):
        # 计算可达距离
        dist = self.get_dist()  # 计算欧氏距离
        # print(dist)
        nei_kdist = np.apply_along_axis(self._kdist, 1, dist)  # 返回每个点的k距离
        # nei_inds, kdist = zip(*nei_kdist)  # 获取前k个距离的索引和k距离
        # enumerate 函数用于遍历序列中的元素以及它们的下标，i为k的下标
        for i, k in enumerate(nei_kdist):
            # 满足条件返回索引
            ind = np.where(dist[:, i] < k)  # 找出i点到其他点的实际距离小于i点k距离的点，将i点到该点的可达距离置为i的k距离
            # print(ind)
            for j in ind:
                dist[j, i] = k
        # print(dist)
        return dist  # 返回前k个距离的索引和每两点之间的可达距离

    def get_lrd(self, rdist):
        # 计算局部可达密度
        dist = self.get_dist()  # 计算欧氏距离
        nei_kdist = np.apply_along_axis(self._kdist, 1, dist)  # 返回每个点的k距离
        # nei_inds, kdist = zip(*nei_kdist)  # 获取前k个距离的索引和k距离
        lrd = np.zeros(self.N)  # 返回一个N个元素的0矩阵
        # 获取每个点的k邻域里包含点的个数，因为距离可能有相等的，所以点的个数应该大于或等于k
        # print(rdist)
        for i, k in enumerate(nei_kdist):
            n = np.where(dist[i] <= k, 1, 0)
            s = np.where(dist[i] <= k)
            # print(s)
            Nk = sum(n) - 1
            sum_rdist = 0
            for j in s:
                sum_rdist = sum(rdist[i][j]) - k
            # print(sum_rdist)
            lrd[i] = Nk/sum_rdist
            # print(lrd)
        return lrd

    def lofk(self):
        # 计算局部离群因子
        dist = self.get_dist()  # 计算欧氏距离
        nei_kdist = np.apply_along_axis(self._kdist, 1, dist)  # 返回每个点的k距离
        rdist = self.get_rdist()  # 获取可达距离
        lrd = self.get_lrd(rdist)  # 获取局部可达密度
        # print(lrd)
        score = np.zeros(self.N)
        for i, k in enumerate(nei_kdist):
            n = np.where(dist[i] <= k, 1, 0)
            s = np.where(dist[i] <= k)
            Nk = sum(n) - 1
            # print(Nk)
            # print(lrd[s])
            lrd_nei = sum(lrd[s]/lrd[i]) - 1
            # print(lrd_nei)
            score[i] = lrd_nei/Nk
            # print(score)
        return np.where(score > self.epsilon)[0], np.where(score < self.epsilon)[0]

    def run(self):
        outliers_ind, inliers_ind = self.lofk()
        outliers = self.data[outliers_ind]
        inliers = self.data[inliers_ind]
        if np.shape(self.data)[1] == 2:
            plt.scatter(self.data[:, 0], self.data[:, 1], color='b')
            plt.scatter(outliers[:, 0], outliers[:, 1], color='r')
        if np.shape(self.data)[1] == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], color='b')
            ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='r')
        plt.show()
        return outliers, inliers


if __name__ == '__main__':
    np.random.seed(42)  # 生成随机数，seed()的作用：使每次生成的随机数相同，若不设置seed()则每次生成的随机数不同
    x_inliers = 0.3 * np.random.randn(100, 2)  # 返回100*2的矩阵
    x_inliers = np.r_[x_inliers + 1, x_inliers - 2]  # 按列连接两个矩阵，就是将两矩阵上下相加，要求列数相等。np_c：按行连接两个矩阵，两矩阵左右相加，要求行数相等
    x_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))  # 从一个均匀分布[low,high)中随机采样，输出200*1个值
    data = np.r_[x_inliers, x_outliers]

    lof = LOF(data, 10, epsilon=1.6)
    outliers, inliers = lof.run()
    print('outliers=', outliers)
    print('inliers=', inliers)
