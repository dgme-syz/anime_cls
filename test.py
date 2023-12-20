import numpy as np
from scipy.stats import anderson
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# def check_nor_at_alpha_anderson(data_, alpha_list=[0.01, 0.05, 0.1]):
#     """
#     输入: 样本数 x 样本特征的二维矩阵(9000 x 12000)
#     输出: 不同显著性水平下通过检验的维度和最低的显著性水平 (使用 Anderson-Darling 测试)
#
#     参数:
#     - data_: 输入的二维矩阵
#     - alpha_list: 一个包含不同显著性水平的列表，默认为 [0.01, 0.05, 0.1]
#
#     返回:
#     - passed_dimensions: 一个字典，键为显著性水平，值为通过检验的维度列表
#     - min_alpha: 最低的显著性水平
#     """
#
#     num_samples, num_features = data_.shape
#     passed_dimensions = {alpha: [] for alpha in alpha_list}
#     min_alpha = None
#
#     for alpha in alpha_list:
#         for feature_idx in range(num_features):
#             # 提取当前维度的数据
#             feature_data = data_[:, feature_idx]
#
#             # 进行 Anderson-Darling 测试
#             result = anderson(feature_data)
#
#             # 检查统计量是否小于临界值，以判断是否通过检验
#             if result.statistic < result.critical_values[2]:
#                 passed_dimensions[alpha].append(feature_idx)
#
#         # 记录最低的显著性水平
#         if not min_alpha or alpha < min_alpha:
#             min_alpha = alpha
#
#     return passed_dimensions, min_alpha
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
        alpha = 0.1
        if p_value < alpha:
            print("拒绝原假设，类别之间存在显著性差异。")
        else:
            print("未拒绝原假设，类别之间没有显著性差异。")
        print("\n")
# 使用示例
# 假设 data 是你的数据矩阵
# passed_dimensions 是一个字典，包含不同显著性水平下通过检验的维度列表
# min_alpha 是最低的显著性水平

start = time.perf_counter()
data = np.random.randn(2500)  # 这只是一个示例，用随机数据代替

# 生成对应的 y 数据，假设有两个类别
y = np.random.choice([0, 1, 2, 3, 4], size=2500)
check_dif(data, y)

# 输出结果

end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))