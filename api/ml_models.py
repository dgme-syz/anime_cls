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
# from scipy.stats import multivariate_normaltest
import seaborn as sns
from pylab import *
from scipy.stats import f_oneway

mpl.rcParams['font.sans-serif'] = ['SimHei']


# # 1. 检验高维数据是否满足正态分布
# def check_nor(data_):
#     """
#     输入: 样本数 x 样本特征的二维矩阵(9000 x 12000)
#     输出: 通过检验的置信度判断是否近似满足正态分布
#     tip: 可以取其中几个具有代表性质的维度展现是否近似满足正态分布
#     """
#     alpha_values = [0.1, 0.21]
#
#     # 进行Shapiro-Wilk检验并观察不同显著性水平下的结果
#     for alpha in alpha_values:
#         statistic, p_value = multivariate_normaltest(data_)
#         print(f"显著性水平为 {alpha} 时的Shapiro-Wilk统计量: {statistic}, p值: {p_value}")
#
#         if p_value >= alpha:
#             print(f"在显著性水平为 {alpha} 下，接受原假设，数据可能在这个水平下近似服从正态分布。\n")
#         else:
#             print(f"在显著性水平为 {alpha} 下，拒绝原假设，数据不在这个水平下近似服从正态分布。\n")


# 2. 检验数据相关性质，绘制相关系数矩阵，用热力图表示
def check_cov(data_):
    """
    输入: 样本数 x 样本特征的二维矩阵(9000 x h)
    输出: 通过绘制相关系数矩阵判断是否有几个维度相关性很大
    tip: 可以取其中几个具有代表性质的维度
    """
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(data_, rowvar=False)
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

    # 绘制直方图
    plt.figure(figsize=(8, 6))
    plt.hist(corr_values, bins=50, alpha=0.7, color='blue')  # 调整 bins 的数量以获得更细或更粗的直方图
    plt.title('相关系数直方图')
    plt.xlabel('相关系数值')
    plt.ylabel('频数')
    plt.grid(True)
    plt.show()


# 3. 为了保证距离判别法的有效性，需要先检验类别是否具有显著性差异
def check_dif(X, y):
    """
    输入: X 每一行为一个样本的特征 y 的每一行为样本的标签
    输出: 在假设每个类别总体近似符合正态分布的前提下，每个类别的均值是否具有显著性差异
    """
    unique_labels = np.unique(y)

    for label in unique_labels:
        # 获取每个类别的样本特征
        features_for_label = X[y == label]

        # 进行方差分析
        f_statistic, p_value = f_oneway(*[X[y == k] for k in unique_labels])

        # 输出结果
        print(f"类别 {label}:")
        print(f"F-statistic: {f_statistic}")
        print(f"P-value: {p_value}")

        # 判断显著性水平（通常使用0.05）来确定是否拒绝原假设
        alpha = 0.05
        if p_value < alpha:
            print("拒绝原假设，类别之间存在显著性差异。")
        else:
            print("未拒绝原假设，类别之间没有显著性差异。")
        print("\n")

"""
tr_X 是一个矩阵, tr_y 是一个 0-8 的数字, 表示标签
"""


def pca_method(tr_X, te_X):
    """
    输入: 训练集的特征 tr_X, 测试集的特征 te_X
    输出: 使用 pca 对于 tr_X 做降维(自行选择一个降维的合适维度) 
          然后使用这个训练好的 pca 对于 te_X 做降维, 输出二者降维后的结果
    """
    # 使用PCA对训练集进行降维
    n_components = 10  # 自定义降维的合适维度
    pca = PCA(n_components=n_components)
    tr_X_pca = pca.fit_transform(tr_X)

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


'''

tr_X = np.array([[1, 2], [3, 4], [5, 6]])
tr_y = np.array([0, 1, 2])
te_X = np.array([[1, 1], [4, 5], [6, 7]])
te_y = np.array([3, 4, 5])

svm_method(tr_X, tr_y, te_X, te_y)
knn_method(tr_X, tr_y, te_X, te_y)
decision_tree_method(tr_X, tr_y, te_X, te_y)

tr_X = np.array([[1, 2], [3, 4], [5, 6]])
tr_y = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
te_X = np.array([[1, 1], [4, 5], [6, 7]])
te_y = np.array([[0.3, 0.7], [0.6, 0.4], [0.8, 0.2]])

multivariables_linear_regression(tr_X, tr_y, te_X, te_y)

'''