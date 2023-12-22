import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from scipy.special import softmax
import seaborn as sns
from pylab import *
import time, matplotlib
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import chi2

# 显示中文字符和负数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择一个包含中文字符的字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
matplotlib.rcParams['font.family'] = 'Arial'

# 1. 检验高维数据是否满足正态分布
def check_nor(data_, labels):
    unique_labels = np.unique(labels)
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))  # 调整图的大小和子图的布局

    # 对每个类别执行操作
    for i, label in enumerate(unique_labels):
        data_class = data_[labels == label]
        num_samples, num_features = data_class.shape
        # print(data_class[0].shape)
        mean_vector = np.mean(data_class, axis=0).reshape(-1, 1)
        cov_matrix_inv = np.linalg.inv(np.cov(data_class, rowvar=False))
        mahalanobis_distances = []
        for x in data_class:
            x = x.reshape(-1, 1)
            val = (x - mean_vector).T @ cov_matrix_inv @ (x - mean_vector)
            mahalanobis_distances.append(val[0][0])
        # print('m', mahalanobis_distances)
        sorted_distances = sorted(mahalanobis_distances)

        p_t = (np.arange(1, num_samples + 1) - 0.5) / num_samples
        chi_square_t = chi2.ppf(p_t, df=num_features)

        assert num_features == 100

        a, b = i // 3, i % 3
        axs[a, b].scatter(sorted_distances, chi_square_t, label=f'Class {label}', color='blue', alpha=0.6,
                            marker='o')
        # 添加斜率为1的线
        axs[a, b].plot([0, sorted_distances[-1]], [0, sorted_distances[-1]], color='red', linestyle='--', label='Line')
        axs[a, b].set_xlabel('马氏距离平方')
        axs[a, b].set_ylabel('卡方分布分位数')
        axs[a, b].set_title(f'Class {label}')
        axs[a, b].legend()
        axs[a, b].grid(True)
    plt.tight_layout()
    plt.show()


# 2. 检验数据相关性质，绘制相关系数矩阵，用热力图表示
def check_cov(data_):
    """
    输入: 样本数 x 样本特征的二维矩阵(9000 x h)
    输出: 通过绘制相关系数矩阵判断是否有几个维度相关性很大
    tip: 可以取其中几个具有代表性质的维度
    """
    # 计算相关系数矩阵
    start = time.time()
    corr_matrix = np.corrcoef(data_, rowvar=False)
    end = time.time()
    print(f"times: {end - start} secs")
    print(corr_matrix.shape)
    num_features_to_select = 10
    num_total_features = data_.shape[1]
    selected_feature_indices = np.random.choice(num_total_features, num_features_to_select, replace=False)

    # 选取对应的子矩阵
    selected_cov_matrix = corr_matrix[selected_feature_indices][:, selected_feature_indices]

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(selected_cov_matrix, cmap='coolwarm', annot=True, fmt=".2f", cbar_kws={'label': 'Covariance'})
    plt.title('Covariance Matrix Heatmap for 10 Random Features')
    plt.show()
    # 将相关系数矩阵展平为一维数组
    flatten_corr = corr_matrix.flatten()

    # 统计相关系数的值
    corr_values = np.sort(flatten_corr)  # 对相关系数值进行排序
    large_cor = np.sum(corr_values >= 0.7)
    print(f"强相关( >= 0.7) 有 {large_cor} 对特征对")
    # 绘制直方图
    plt.figure(figsize=(8, 6))
    sns.histplot(corr_values, bins=20, kde=True, stat='count', \
                 line_kws={'color': '#a7f2a7', 'lw': 3, 'ls': ':'}) 

    # 添加标题和标签
    plt.title('相关系数直方图')
    plt.xlabel('相关系数值')
    plt.ylabel('频数')

    # 显示图例
    plt.legend(["KDE曲线"])
    plt.grid()
    plt.savefig('./img/1.png', dpi=600)
    plt.show()


# 3. 为了保证距离判别法的有效性，需要先检验类别是否具有显著性差异
def check_dif(X, y):
    """
    输入: X 每一行为一个样本的特征 y 的每一行为样本的标签
    输出: 在假设每个类别总体近似符合正态分布的前提下，每个类别的均值是否具有显著性差异
    """
    unique_labels = np.unique(y)

    for label in unique_labels:
        # 获取当前类别的样本特征
        features_for_label = X[y == label]

        # 进行方差分析，这里使用当前类别的数据
        f_statistic, p_value = f_oneway(*[features_for_label, *[X[y == k] for k in unique_labels if k != label]])

        # 输出结果
        print(f"类别 {label}:")
        print(f"F-statistic: {f_statistic}")
        print(f"P-value: {p_value}")

        # 判断显著性水平（通常使用0.05）来确定是否拒绝原假设
        alpha = 0.05
        if any(p_value < alpha):
            print("拒绝原假设，类别之间存在显著性差异。")
        else:
            print("未拒绝原假设，类别之间没有显著性差异。")
        print("\n")

"""
tr_X 是一个矩阵, tr_y 是一个 0-8 的数字, 表示标签
"""


def pca_method(tr_X, te_X, decompose=100):
    """
    输入: 训练集的特征 tr_X, 测试集的特征 te_X
    输出: 使用 pca 对于 tr_X 做降维(自行选择一个降维的合适维度) 
          然后使用这个训练好的 pca 对于 te_X 做降维, 输出二者降维后的结果
    """
    # 使用PCA对训练集进行降维
    n_components = decompose  # 自定义降维的合适维度
    pca = PCA(n_components=n_components)
    tr_X_pca = pca.fit_transform(tr_X)
    total_explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"整体方差解释比例: {total_explained_variance_ratio}")
    # 使用训练好的PCA对测试集进行降维
    te_X_pca = pca.transform(te_X)
    return tr_X_pca, te_X_pca


def ker_pca_method(tr_X, te_X):
    """
    输入: 训练集的特征 tr_X, 测试集的特征 te_X
    输出: 使用 tanh核pca 对于 tr_X 做降维(自行选择一个降维的合适维度) 
          然后使用这个训练好的 pca 对于 te_X 做降维, 输出二者降维后的结果
    """
    # 使用tanh核PCA对训练集进行降维
    n_components = 10  # 自定义降维的合适维度
    kpca = KernelPCA(n_components=n_components, kernel='sigmoid')
    tr_X_kpca = kpca.fit_transform(tr_X)

    # 使用训练好的PCA对测试集进行降维
    te_X_kpca = kpca.transform(te_X)

    return tr_X_kpca, te_X_kpca


def flda_method(tr_X, tr_y, te_X):
    """
    输入: 训练集的特征 tr_X, 测试集的特征 te_X
    输出: 使用 FLDA 对于 tr_X 以及 tr_y 做降维(自行选择一个降维的合适维度)
          然后使用这个训练好的 pca 对于 te_X 做降维, 输出二者降维后的结果
    """
    # 使用FLDA对训练集进行降维
    flda = LinearDiscriminantAnalysis(n_components=1)  # 自行选择合适的降维维度
    tr_X_flda = flda.fit_transform(tr_X, tr_y)

    # 使用训练好的FLDA模型对测试集进行降维
    te_X_flda = flda.transform(te_X)

    return tr_X_flda, te_X_flda


def decision_tree_method(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y) 训练决策树
    输出: 测试集上的 acc,以及整体的混淆矩阵,需要可视化
    """
    # 训练决策树模型
    clf = DecisionTreeClassifier()
    clf.fit(tr_X, tr_y)

    # 在测试集上进行预测
    te_pred = clf.predict(te_X)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    print('测试集准确率:', acc)

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()


def multivariables_linear_regression(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, softmax(tr_y)) 训练多因变量线性回归
    输出: 测试集上的 acc,以及整体的混淆矩阵,需要可视化
    """
    # 对训练集和测试集的标签进行 softmax 转换
    tr_y_softmax = softmax(tr_y, axis=1)
    te_y_softmax = softmax(te_y, axis=1)

    # 训练多变量线性回归模型
    regression = LinearRegression()
    regression.fit(tr_X, tr_y_softmax)

    # 在测试集上进行预测
    te_pred_softmax = regression.predict(te_X)
    te_pred = np.argmax(te_pred_softmax, axis=1)

    # 计算准确率
    acc = accuracy_score(np.argmax(te_y_softmax, axis=1), te_pred)
    print('测试集准确率:', acc)

    # 计算混淆矩阵
    cm = confusion_matrix(np.argmax(te_y_softmax, axis=1), te_pred)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def knn_method(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y), 使用 knn 模型, 近邻数目自己调参
    输出: 测试集上的 acc,以及整体的混淆矩阵,需要可视化
    """
    n_neighbors = 3  # 设置近邻数目为3

    # 训练 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(tr_X, tr_y)

    # 在测试集上进行预测
    te_pred = knn.predict(te_X)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    print('测试集准确率:', acc)

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def svm_method(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y), 使用 tanh 核 svm 训练
    输出: 测试集上的 acc,以及整体的混淆矩阵,需要可视化
    """
    # 训练 SVM 模型
    svm = SVC(kernel='sigmoid')
    svm.fit(tr_X, tr_y)

    # 在测试集上进行预测
    te_pred = svm.predict(te_X)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    print('测试集准确率:', acc)

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def collect_data(train_data, test_data):
    G = [[] for i in range(len(np.unique(test_data)))]
    for i in range(len(test_data)):
        G[test_data[i]].append(train_data[i])
    return G

def TwoScatter(train_data, test_data):
    """
    用密度曲线可视化二维数据
    """
    color = "flare"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    
    # 隐藏标签
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    # 散点图与密度曲线
    n = len(np.unique(test_data))
    x, y = train_data[:, 0], train_data[:, 1]
    # 绘制散点图，颜色根据 test_data 指定
    sns.scatterplot(x=x, y=y, hue=test_data, ax=ax, palette=color)
    # 收集数据
    cls = collect_data(train_data, test_data)

    for i in range(n):
        # x轴方向的密度曲线
        xx = [a for a, b in cls[i]]
        sns.kdeplot(pd.DataFrame({'x': xx}), ax=ax_histx, fill=True, x='x', \
                    color=sns.color_palette(color, n)[i])
        
        # y轴方向的密度曲线
        yy = [b for a, b in cls[i]]
        sns.kdeplot(pd.DataFrame({'y': yy}), ax=ax_histy, fill=True, y='y', \
                    color=sns.color_palette(color, n)[i])

    ax.grid()
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax_histx.grid()
    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    ax_histx.set_xlabel('')
    ax_histx.set_ylabel('')
    ax_histy.set_xlabel('')
    ax_histy.set_ylabel('')
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.grid()
    ax_histy.grid(alpha=0.5, linewidth=0.5)

    # 显示图例
    ax.legend()
    plt.show()
if __name__ == '__main__':
    np.random.seed(19680801)

# some random data
    x = 100 * np.random.randn(1000, 2)
    y = np.random.randint(9, size=1000)
    # print(y)
    TwoScatter(x, y)