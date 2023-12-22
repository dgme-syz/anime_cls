from scipy.stats import chi2
from matplotlib.colors import ListedColormap
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import seaborn as sns
from pylab import *
<<<<<<< HEAD


mpl.rcParams['font.sans-serif'] = ['SimHei']
=======
import time, matplotlib
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import chi2
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
>>>>>>> 706aefbe8b659b26ec2be7304016f52801107e69

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
        mean_vector = np.mean(data_class, axis=0)
        cov_matrix = np.cov(data_class, rowvar=False)
        mahalanobis_distances = np.array(
            [np.dot(np.dot((x - mean_vector).T, np.linalg.inv(cov_matrix)), (x - mean_vector)) for x in
             data_class])

        sorted_distances_indices = np.argsort(mahalanobis_distances)
        sorted_distances = mahalanobis_distances[sorted_distances_indices]

        p_t = (np.arange(1, num_samples + 1) - 0.5) / num_samples
        chi_square_t = chi2.ppf(p_t, df=num_features)

        # 使用索引选择正确的子图进行绘制
        row_index = i // 3
        col_index = i % 3

        # 定义散点大小
        scatter_size = 10  # 调整这个值以改变散点的大小

        # 自定义颜色映射
        custom_colors = ['#158bb8', '#0eb0c9', '#51c4d3', '#83cbac', '#55bb8a']
        custom_cmap = ListedColormap(custom_colors)

        # 使用自定义颜色映射
        colors = custom_cmap(np.linspace(0, 1, num_samples))
        # 这里使用颜色映射，可以根据某个特征值的范围为每个数据点分配颜色

        axs[row_index, col_index].scatter(
            sorted_distances,
            chi_square_t,
            label=f'类别 {label}',
            color=colors,  # 使用颜色映射
            alpha=0.6,
            marker='o',
            s=scatter_size
        )

        unique_distances, unique_indices = np.unique(sorted_distances, return_index=True)
        axs[row_index, col_index].plot(unique_distances, sorted_distances[unique_indices], color='#207f4c', linestyle='-.')
        np.random.seed(42)
        x = np.linspace(min(unique_distances), max(unique_distances), 1000)
        y1 = x+3 - x / 8 + np.random.uniform(0.0, 0.5, len(x))
        y2 = x+3 + x / 8 + np.random.uniform(0.0, 0.5, len(x))
        axs[row_index, col_index].fill_between(x, y1, y2, alpha=.5, linewidth=0, color='#add5a2')

        # 设置轴标签和标题
        axs[row_index, col_index].set_xlabel('马氏距离')
        axs[row_index, col_index].set_ylabel('卡方分位数')
        axs[row_index, col_index].set_title(f'类别 {label}: {num_samples} 个数据')

        # 显示图例
        axs[row_index, col_index].legend()

        # 显示网格
        axs[row_index, col_index].grid(True, alpha=0.1)
        for spine in axs[row_index, col_index].spines.values():
            spine.set_edgecolor('#add5a2')
            spine.set_alpha(0.5)

    plt.tight_layout()
    fig.savefig('Q-Q.png', dpi=600)
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


"""
tr_X 是一个矩阵, tr_y 是一个 0-8 的数字, 表示标签
"""


def pca_method(tr_X, te_X, n_components = 100):
    """
    输入: 训练集的特征 tr_X, 测试集的特征 te_X
    输出: 使用 pca 对于 tr_X 做降维(自行选择一个降维的合适维度)
          然后使用这个训练好的 pca 对于 te_X 做降维, 输出二者降维后的结果
    """
    # 使用PCA对训练集进行降维
    pca = PCA(n_components=n_components)
    tr_X_pca = pca.fit_transform(tr_X)

    # 使用训练好的PCA对测试集进行降维
    te_X_pca = pca.transform(te_X)

    # 输出方差解释率
    explained_variance_ratio = pca.explained_variance_ratio_
    variance_ratio_sum = np.sum(explained_variance_ratio)
    print("PCA方差解释率:", variance_ratio_sum)

    return tr_X_pca, te_X_pca, variance_ratio_sum


'''
def ker_pca_method(tr_X, te_X):
    """
    输入: 训练集的特征 tr_X, 测试集的特征 te_X
    输出: 使用 tanh核pca 对于 tr_X 做降维(自行选择一个降维的合适维度) 
          然后使用这个训练好的 pca 对于 te_X 做降维, 输出二者降维后的结果
    """
    # 使用tanh核PCA对训练集进行降维
    n_components = 100  # 自定义降维的合适维度
    kpca = KernelPCA(n_components=n_components, kernel='sigmoid')
    tr_X_kpca = kpca.fit_transform(tr_X)

    # 使用训练好的PCA对测试集进行降维
    te_X_kpca = kpca.transform(te_X)

    return tr_X_kpca, te_X_kpca
'''


def flda_method(tr_X, tr_y, te_X, decompose=8):
    """
    输入: 训练集的特征 tr_X, 测试集的特征 te_X
    输出: 使用 FLDA 对于 tr_X 以及 tr_y 做降维(自行选择一个降维的合适维度)  
          然后使用这个训练好的 pca 对于 te_X 做降维, 输出二者降维后的结果
    """
    # 使用FLDA对训练集进行降维
<<<<<<< HEAD
    flda = LinearDiscriminantAnalysis(n_components=8)  # 自行选择合适的降维维度
=======
    flda = LinearDiscriminantAnalysis(n_components=decompose)  # 自行选择合适的降维维度
>>>>>>> 706aefbe8b659b26ec2be7304016f52801107e69
    tr_X_flda = flda.fit_transform(tr_X, tr_y)

    # 使用训练好的FLDA模型对测试集进行降维
    te_X_flda = flda.transform(te_X)

    return tr_X_flda, te_X_flda


def decision_tree_method(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y) 训练决策树
    输出: 测试集上的 acc,以及整体的混淆矩阵,需要可视化
    """
    # 记录开始时间
    start_time = time.time()

    # 获取输入矩阵的列数
    num_columns = tr_X.shape[1]

    # 训练决策树模型
    clf = DecisionTreeClassifier()
    clf.fit(tr_X, tr_y)

    # 在测试集上进行预测
    te_pred = clf.predict(te_X)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    acc_name = "决策树在"
    acc_name += str(num_columns) + '维时测试集准确率为:'
    print(acc_name, acc)

    # 计算Micro F1
    micro_f1 = f1_score(te_y, te_pred, average='micro')

    # 计算Macro F1
    macro_f1 = f1_score(te_y, te_pred, average='macro')

    # 计算最小召回率
    min_recall = np.min(recall_score(te_y, te_pred, average=None))

    # 计算每个类别的准确率
    te_y = np.array(te_y)
    unique_labels = np.unique(te_y)
    class_acc = {}
    for label in unique_labels:
        class_indices = np.where(te_y == label)[0]
        class_acc[label] = accuracy_score(te_y[class_indices], te_pred[class_indices])
    # 计算每个类别的准确率的平均值
    avg_class_acc = np.mean(list(class_acc.values()))

    # 计算每个类别的准确率的调和平均值
    harmonic_mean = len(class_acc) / np.sum(1 / np.array(list(class_acc.values())))

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    run_time = end_time - start_time

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    # 设置混淆矩阵标题
    title_name = '决策树混淆矩阵（降维至'
    title_name += str(num_columns) + '维时）'

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues')
    plt.title(title_name)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 定义保存路径和文件名
    save_dir = './result_fig'
    save_filename = 'dtree_cm_'

    # 将列数与文件名拼接
    save_filename += str(num_columns) + '.png'

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图形，并指定 DPI
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=600)
    plt.close()

    # 提示保存成功
    print(f"决策树图像已保存至: {save_path}")

    return acc, run_time, micro_f1, macro_f1, min_recall, avg_class_acc, harmonic_mean


def multivariables_linear_regression(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y) 训练多因变量线性回归
    输出: 测试集上的准确率 acc，以及整体的混淆矩阵，并进行可视化
    """
    # 记录开始时间
    start_time = time.time()

    # 获取输入矩阵的列数
    num_columns = tr_X.shape[1]

    # 对训练集和测试集的标签进行 One-Hot 编码
    num_classes = 9
    tr_y_onehot = np.eye(num_classes)[tr_y]

    # 训练多变量线性回归模型
    regression = LinearRegression()
    regression.fit(tr_X, tr_y_onehot)

    # 在测试集上进行预测
    te_pred_onehot = regression.predict(te_X)
    te_pred = np.argmax(te_pred_onehot, axis=1)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    acc_name = "多因变量线性回归在"
    acc_name += str(num_columns) + '维时测试集准确率为:'
    print(acc_name, acc)

    # 计算Micro F1
    micro_f1 = f1_score(te_y, te_pred, average='micro')
    # 计算Macro F1
    macro_f1 = f1_score(te_y, te_pred, average='macro')

    # 计算最小召回率
    min_recall = np.min(recall_score(te_y, te_pred, average=None))

    # 计算每个类别的准确率
    te_y = np.array(te_y)
    unique_labels = np.unique(te_y)
    class_acc = {}
    for label in unique_labels:
        class_indices = np.where(te_y == label)[0]
        class_acc[label] = accuracy_score(te_y[class_indices], te_pred[class_indices])

    # 计算每个类别的准确率的平均值
    avg_class_acc = np.mean(list(class_acc.values()))

    # 计算每个类别的准确率的调和平均值
    harmonic_mean = len(class_acc) / np.sum(1 / np.array(list(class_acc.values())))

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    run_time = end_time - start_time

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    # 设置混淆矩阵标题
    title_name = '多因变量线性回归混淆矩阵（降维至'
    title_name += str(num_columns) + '维时）'

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues')
    plt.title(title_name)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 定义保存路径和文件名
    save_dir = './result_fig'
    save_filename = 'mlr_cm_'

    # 将列数与文件名拼接
    save_filename += str(num_columns) + '.png'

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图形，并指定 DPI
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=600)
    plt.close()

    # 提示保存成功
    print(f"多因变量线性回归图像已保存至: {save_path}")

    return acc, run_time, micro_f1, macro_f1, min_recall, avg_class_acc, harmonic_mean


def knn_method(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y), 使用 knn 模型, 近邻数目自己调参
    输出: 测试集上的 acc,以及整体的混淆矩阵,需要可视化
    """
    # 记录开始时间
    start_time = time.time()

    # 获取输入矩阵的列数
    num_columns = tr_X.shape[1]

    n_neighbors = 5  # 设置近邻数目为5

    # 训练 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(tr_X, tr_y)

    # 在测试集上进行预测
    te_pred = knn.predict(te_X)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    acc_name = "KNN在"
    acc_name += str(num_columns) + '维时测试集准确率为:'
    print(acc_name, acc)

    # 计算Micro F1
    micro_f1 = f1_score(te_y, te_pred, average='micro')

    # 计算Macro F1
    macro_f1 = f1_score(te_y, te_pred, average='macro')

    # 计算最小召回率
    min_recall = np.min(recall_score(te_y, te_pred, average=None))

    # 计算每个类别的准确率
    te_y = np.array(te_y)
    unique_labels = np.unique(te_y)
    class_acc = {}
    for label in unique_labels:
        class_indices = np.where(te_y == label)[0]
        class_acc[label] = accuracy_score(te_y[class_indices], te_pred[class_indices])

    # 计算每个类别的准确率的平均值
    avg_class_acc = np.mean(list(class_acc.values()))

    # 计算每个类别的准确率的调和平均值
    harmonic_mean = len(class_acc) / np.sum(1 / np.array(list(class_acc.values())))

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    run_time = end_time - start_time

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    # 设置混淆矩阵标题
    title_name = 'knn混淆矩阵（降维至'
    title_name += str(num_columns) + '维时）'

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues')
    plt.title(title_name)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 定义保存路径和文件名
    save_dir = './result_fig'
    save_filename = 'knn_cm_'

    # 将列数与文件名拼接
    save_filename += str(num_columns) + '.png'

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图形，并指定 DPI
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=600)
    plt.close()

    # 提示保存成功
    print(f"knn图像已保存至: {save_path}")

    return acc, run_time, micro_f1, macro_f1, min_recall, avg_class_acc, harmonic_mean


def svm_method(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y), 使用 tanh 核 svm 训练
    输出: 测试集上的 acc,以及整体的混淆矩阵,需要可视化
    """
    # 记录开始时间
    start_time = time.time()

    # 获取输入矩阵的列数
    num_columns = tr_X.shape[1]

    # 训练 SVM 模型
    svm = SVC(kernel='rbf')
    svm.fit(tr_X, tr_y)

    # 在测试集上进行预测
    te_pred = svm.predict(te_X)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    acc_name = "SVM在"
    acc_name += str(num_columns) + '维时测试集准确率为:'
    print(acc_name, acc)

    # 计算Micro F1
    micro_f1 = f1_score(te_y, te_pred, average='micro')

    # 计算Macro F1
    macro_f1 = f1_score(te_y, te_pred, average='macro')

    # 计算最小召回率
    min_recall = np.min(recall_score(te_y, te_pred, average=None))

    # 计算每个类别的准确率
    te_y = np.array(te_y)
    unique_labels = np.unique(te_y)
    class_acc = {}
    for label in unique_labels:
        class_indices = np.where(te_y == label)[0]
        class_acc[label] = accuracy_score(te_y[class_indices], te_pred[class_indices])

    # 计算每个类别的准确率的平均值
    avg_class_acc = np.mean(list(class_acc.values()))

    # 计算每个类别的准确率的调和平均值
    harmonic_mean = len(class_acc) / np.sum(1 / np.array(list(class_acc.values())))

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    run_time = end_time - start_time

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    # 设置混淆矩阵标题
    title_name = 'svm混淆矩阵（降维至'
    title_name += str(num_columns) + '维时）'

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues')
    plt.title(title_name)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 定义保存路径和文件名
    save_dir = './result_fig'
    save_filename = 'svm_cm_'

    # 将列数与文件名拼接
    save_filename += str(num_columns) + '.png'

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图形，并指定 DPI
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=600)
    plt.close()

    # 提示保存成功
    print(f"svm图像已保存至: {save_path}")

    return acc, run_time, micro_f1, macro_f1, min_recall, avg_class_acc, harmonic_mean

def collect_data(train_data, test_data):
    G = [[] for i in range(len(np.unique(test_data)))]
    for i in range(len(test_data)):
        G[test_data[i]].append(train_data[i])
    return G

def qda_method(tr_X, tr_y, te_X, te_y):
    """
    输入: 训练集 (tr_X, tr_y) 和测试集 (te_X, te_y)
    输出: 测试集上的准确率 (acc) 和混淆矩阵 (cm)，并可视化混淆矩阵
    返回: 准确率 (acc) 和运行时间 (run_time)
    """
    # 记录开始时间
    start_time = time.time()

    # 获取输入矩阵的列数
    num_columns = tr_X.shape[1]

    # 创建QDA分类器对象
    qda = QuadraticDiscriminantAnalysis()

    # 在训练集上训练QDA分类器
    qda.fit(tr_X, tr_y)

    # 在测试集上进行预测
    te_pred = qda.predict(te_X)

    # 计算准确率
    acc = accuracy_score(te_y, te_pred)
    acc_name = "QDA在"
    acc_name += str(num_columns) + '维时测试集准确率为:'
    print(acc_name, acc)

    # 计算Micro F1
    micro_f1 = f1_score(te_y, te_pred, average='micro')

    # 计算Macro F1
    macro_f1 = f1_score(te_y, te_pred, average='macro')

    # 计算最小召回率
    min_recall = np.min(recall_score(te_y, te_pred, average=None))

    # 计算每个类别的准确率
    te_y = np.array(te_y)
    unique_labels = np.unique(te_y)
    class_acc = {}
    for label in unique_labels:
        class_indices = np.where(te_y == label)[0]
        class_acc[label] = accuracy_score(te_y[class_indices], te_pred[class_indices])

    # 计算每个类别的准确率的平均值
    avg_class_acc = np.mean(list(class_acc.values()))

    # 计算每个类别的准确率的调和平均值
    harmonic_mean = len(class_acc) / np.sum(1 / np.array(list(class_acc.values())))

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    run_time = end_time - start_time

    # 计算混淆矩阵
    cm = confusion_matrix(te_y, te_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    # 设置混淆矩阵标题
    title_name = 'QDA混淆矩阵（降维至'
    title_name += str(num_columns) + '维时）'

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues')
    plt.title(title_name)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 定义保存路径和文件名
    save_dir = './result_fig'
    save_filename = 'qda_cm_'

    # 将列数与文件名拼接
    save_filename += str(num_columns) + '.png'

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图形，并指定 DPI
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=600)
    plt.close()

    # 提示保存成功
    print(f"qda图像已保存至: {save_path}")

    return acc, run_time, micro_f1, macro_f1, min_recall, avg_class_acc, harmonic_mean

def TwoScatter(train_data, test_data, name='picture'):
    """
    用密度曲线可视化二维数据
    """
    n = len(np.unique(test_data))
    color_map = sns.color_palette("Spectral", n_colors=n)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    
    # 隐藏标签
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    # 散点图与密度曲线
    x, y = train_data[:, 0], train_data[:, 1]
    # 绘制散点图，颜色根据 test_data 指定
    sns.scatterplot(x=x, y=y, hue=test_data, ax=ax, palette=color_map)
    # 收集数据
    cls = collect_data(train_data, test_data)

    for i in range(n):
        # x轴方向的密度曲线
        xx = [a for a, b in cls[i]]
        sns.kdeplot(pd.DataFrame({'x': xx}), ax=ax_histx, fill=True, x='x', \
                    color=color_map[i])
        
        # y轴方向的密度曲线
        yy = [b for a, b in cls[i]]
        sns.kdeplot(pd.DataFrame({'y': yy}), ax=ax_histy, fill=True, y='y', \
                    color=color_map[i])

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
    plt.savefig(f"./img/{name}.png", dpi=600)
    plt.show()
if __name__ == '__main__':
    np.random.seed(19680801)

# some random data
    x = 10000 * np.random.randn(1000, 2)
    y = np.random.randint(9, size=1000)
    # print(np.unique(y))
    TwoScatter(x, y)

