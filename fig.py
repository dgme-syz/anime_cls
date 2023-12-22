import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.special import softmax
from scipy.stats import anderson
import seaborn as sns
from pylab import *
from scipy.stats import f_oneway

# 设置中文字体为宋体（SimSun）
mpl.rcParams['font.sans-serif'] = ['SimSun']

# 设置英文字体为 Times New Roman
mpl.rcParams['font.serif'] = ['Times New Roman']

n_list = [10,50,100,200,500]
n_list_str = ['10', '50', '100', '200', '500']

list_names = ['dtree_acc', 'dtree_time', 'dtree_mif', 'dtree_maf', 'dtree_mrc', 'dtree_aca', 'dtree_hm',
'mlr_acc', 'mlr_time', 'mlr_mif', 'mlr_maf', 'mlr_mrc', 'mlr_aca', 'mlr_hm',
'knn_acc', 'knn_time', 'knn_mif', 'knn_maf', 'knn_mrc', 'knn_aca', 'knn_hm',
'svm_acc', 'svm_time', 'svm_mif', 'svm_maf', 'svm_mrc', 'svm_aca', 'svm_hm',
'qda_acc', 'qda_time', 'qda_mif', 'qda_maf', 'qda_mrc', 'qda_aca', 'qda_hm']
# 创建空列表的字典
lists = {name: [] for name in list_names}

lists['qda_acc'] = [0.7368996827480582, 0.8688327316486161, 0.8557050650913467, 0.8362323596980636, 0.7723443824526857]
lists['qda_time'] = [0.03667926788330078, 0.10614800453186035, 0.19920778274536133, 0.4336252212524414, 1.6547698974609375]
lists['qda_mif'] = [0.7368996827480582, 0.8688327316486161, 0.8557050650913467, 0.8362323596980636, 0.7723443824526857]
lists['qda_maf'] = [0.5012250439603791, 0.713802564479971, 0.6706349247690075, 0.643031778687134, 0.5586340337029685]
lists['qda_mrc'] = [0.045454545454545456, 0.1951219512195122, 0.032520325203252036, 0.008130081300813009, 0.0]
lists['qda_aca'] = [0.518914911522216, 0.708761905870388, 0.6692741758600392, 0.6468424798287871, 0.5448718030560072]
lists['qda_hm'] = [0.1887829510745575, 0.5033576686312177, 0.19721415715509333, 0.06022166669067989, 0.0]

lists['svm_acc'] = [0.5043211902417679, 0.6108740837982716, 0.6419428946504758, 0.6160157531998687, 0.6314407614046603]
lists['svm_time'] = [33.4596061706543, 32.41331744194031, 39.886908531188965, 78.2080819606781, 166.1298372745514]
lists['svm_mif'] = [0.5043211902417679, 0.6108740837982716, 0.6419428946504758, 0.6160157531998687, 0.6314407614046603]
lists['svm_maf'] = [0.3501130395542455, 0.44439045126923243, 0.4829738878262852, 0.4669330968771485, 0.4769786778067247]
lists['svm_mrc'] = [0.09319664492078285, 0.2423112767940354, 0.35855263157894735, 0.2851817334575955, 0.2898415657036347]
lists['svm_aca'] = [0.38281728114013847, 0.47974394356431815, 0.5146701343434112, 0.5045985270577752, 0.5162922242602535]
lists['svm_hm'] = [0.2588039592080649, 0.40751428038771326, 0.47429401791046893, 0.4613994089145398, 0.46739274307229134]

lists['knn_acc'] = [0.6994858330598402, 0.7643583852970135, 0.7591073186741056, 0.7603106881085221, 0.7563723881413412]
lists['knn_time'] = [1.2366492748260498, 1.0063893795013428, 1.1670219898223877, 2.050307035446167, 4.523566722869873]
lists['knn_mif'] = [0.6994858330598402, 0.7643583852970134, 0.7591073186741055, 0.7603106881085222, 0.7563723881413412]
lists['knn_maf'] = [0.47212193340558617, 0.5658178822225579, 0.5565853918192611, 0.5540231870245569, 0.5529679239636334]
lists['knn_mrc'] = [0.0975609756097561, 0.17045454545454544, 0.13636363636363635, 0.11363636363636363, 0.11363636363636363]
lists['knn_aca'] = [0.4742357536849425, 0.5573696694832969, 0.5472649794862511, 0.5434210755138378, 0.5392613798928222]
lists['knn_hm'] = [0.28454960222914355, 0.4033895159064232, 0.3561762314600489, 0.3311942845236661, 0.33410193681414874]

lists['mlr_acc'] = [0.6960945191992124, 0.7759544907559348, 0.7909419100754841, 0.8292309375341866, 0.860299748386391]
lists['mlr_time'] = [0.042108774185180664, 0.06114053726196289, 0.09735488891601562, 0.1612255573272705, 0.4719123840332031]
lists['mlr_mif'] = [0.6960945191992124, 0.7759544907559348, 0.7909419100754841, 0.8292309375341866, 0.860299748386391]
lists['mlr_maf'] = [0.36914841309532104, 0.5316831221855493, 0.5515223283298843, 0.602693806971061, 0.6554821682540041]
lists['mlr_mrc'] = [0.0, 0.0, 0.0, 0.0, 0.0]
lists['mlr_aca'] = [0.3839968448970809, 0.5176536526316666, 0.5409998805298081, 0.5976125084443651, 0.639603608298]
lists['mlr_hm'] = [0.0, 0.0, 0.0, 0.0, 0.0]

lists['dtree_acc'] = [0.6629471611421069, 0.6596652445027896, 0.6524450278962914, 0.644240236297998, 0.625533311453889]
lists['dtree_time'] = [1.0394680500030518, 6.286511659622192, 13.191465854644775, 28.69132113456726, 74.73052263259888]
lists['dtree_mif'] = [0.6629471611421069, 0.6596652445027896, 0.6524450278962914, 0.644240236297998, 0.625533311453889]
lists['dtree_maf'] = [0.43872016134396297, 0.45981244999574256, 0.4454130491987727, 0.43608871542317024, 0.42453703916505603]
lists['dtree_mrc'] = [0.07954545454545454, 0.1590909090909091, 0.14772727272727273, 0.10227272727272728, 0.11363636363636363]
lists['dtree_aca'] = [0.44641200683709237, 0.47410124942614945, 0.45843514304457045, 0.44497661408797484, 0.4414734138143873]
lists['dtree_hm'] = [0.27502032823455946, 0.3646047889825302, 0.34818653952714124, 0.3026986041375434, 0.32590113057221953]

# 绘制各个分类器下不同指标的变化图

line_colors = ['#5890da', '#5890da', '#87cfa4', '#fee8a6', '#469eb4', '#cbe99d']

# 决策树分类器
plot_dtree = ['dtree_acc', 'dtree_mif', 'dtree_maf', 'dtree_mrc', 'dtree_aca', 'dtree_hm']
fig, ax = plt.subplots()
for i, var_name in enumerate(plot_dtree):
    # 获取对应变量的值
    values = lists[var_name]

    # 计算误差带的上下界
    lower_bound = np.array(values) * 0.95
    upper_bound = np.array(values) * 1.05

    # 绘制误差带
    ax.fill_between(n_list_str, lower_bound, upper_bound, alpha=0.2, color=line_colors[i])

    # 绘制折线图，并设置线条颜色为line_colors中的对应颜色
    ax.plot(n_list_str, values, label=var_name, linestyle='-', color=line_colors[i])

    for x, y in zip(n_list_str, values):
        ax.annotate('{:.2f}'.format(y), (x, y), xytext=(0, 0), textcoords='offset points', ha='center', va='bottom', fontsize=6)

# 获取图例的handles和labels
handles, labels = ax.get_legend_handles_labels()

# 创建新的handles和labels，只包含要显示的变量名
new_handles = []
new_labels = []

for handle, var_name in zip(handles, plot_dtree):
    if var_name in plot_dtree:
        new_handles.append(handle)
        if var_name == 'dtree_acc':
            new_labels.append('Acc')
        elif var_name == 'dtree_mif':
            new_labels.append('Micro_F1')
        elif var_name == 'dtree_maf':
            new_labels.append('Macro_F1')
        elif var_name == 'dtree_mrc':
            new_labels.append('LR')
        elif var_name == 'dtree_aca':
            new_labels.append('GM_Acc')
        elif var_name == 'dtree_hm':
            new_labels.append('HM_Acc')

legend = ax.legend(new_handles, new_labels, bbox_to_anchor=(1.0, 1.0), loc='upper right')

ax.set_ylim(0, 1.1)
ax.set_xlabel('N_Components')
ax.set_ylabel('Value')
ax.set_title(r'$\mathbf{Decision\ Tree}$')
# 启用网格线，设置虚线样式，不透明度为0.25
ax.grid(True, linestyle=':', alpha=0.50)
# 去掉四个边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# 显示图形
fig.savefig("./result_fig/dtree_pca.png", dpi=2000)
plt.close()





# 多因变量线性回归分类器
plot_mlr = ['mlr_acc', 'mlr_mif', 'mlr_maf', 'mlr_mrc', 'mlr_aca', 'mlr_hm']
fig, ax = plt.subplots()
for i, var_name in enumerate(plot_mlr):
    # 获取对应变量的值
    values = lists[var_name]

    # 计算误差带的上下界
    lower_bound = np.array(values) * 0.95
    upper_bound = np.array(values) * 1.05

    # 绘制误差带
    ax.fill_between(n_list_str, lower_bound, upper_bound, alpha=0.2, color=line_colors[i])

    # 绘制折线图，并设置线条颜色为line_colors中的对应颜色
    ax.plot(n_list_str, values, label=var_name, linestyle='-', color=line_colors[i])

    for x, y in zip(n_list_str, values):
        ax.annotate('{:.2f}'.format(y), (x, y), xytext=(0, 0), textcoords='offset points', ha='center', va='bottom', fontsize=6)

# 获取图例的handles和labels
handles, labels = ax.get_legend_handles_labels()

# 创建新的handles和labels，只包含要显示的变量名
new_handles = []
new_labels = []

for handle, var_name in zip(handles, plot_mlr):
    if var_name in plot_mlr:
        new_handles.append(handle)
        if var_name == 'mlr_acc':
            new_labels.append('Acc')
        elif var_name == 'mlr_mif':
            new_labels.append('Micro_F1')
        elif var_name == 'mlr_maf':
            new_labels.append('Macro_F1')
        elif var_name == 'mlr_mrc':
            new_labels.append('LR')
        elif var_name == 'mlr_aca':
            new_labels.append('GM_Acc')
        elif var_name == 'mlr_hm':
            new_labels.append('HM_Acc')

legend = ax.legend(new_handles, new_labels, bbox_to_anchor=(1.0, 1.0), loc='upper right')

ax.set_ylim(0, 1.4)
ax.set_xlabel('N_Components')
ax.set_ylabel('Value')
ax.set_title(r'$\mathbf{Multivariables\ Linear\ Regression}$')
# 启用网格线，设置虚线样式，不透明度为0.25
ax.grid(True, linestyle=':', alpha=0.50)
# 去掉四个边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# 显示图形
fig.savefig("./result_fig/mlr_pca.png", dpi=600)
plt.close()





# KNN分类器
plot_knn = ['knn_acc', 'knn_mif', 'knn_maf', 'knn_mrc', 'knn_aca', 'knn_hm']
fig, ax = plt.subplots()
for i, var_name in enumerate(plot_knn):
    # 获取对应变量的值
    values = lists[var_name]

    # 计算误差带的上下界
    lower_bound = np.array(values) * 0.95
    upper_bound = np.array(values) * 1.05

    # 绘制误差带
    ax.fill_between(n_list_str, lower_bound, upper_bound, alpha=0.2, color=line_colors[i])

    # 绘制折线图，并设置线条颜色为line_colors中的对应颜色
    ax.plot(n_list_str, values, label=var_name, linestyle='-', color=line_colors[i])

    for x, y in zip(n_list_str, values):
        ax.annotate('{:.2f}'.format(y), (x, y), xytext=(0, 0), textcoords='offset points', ha='center', va='bottom', fontsize=6)

# 获取图例的handles和labels
handles, labels = ax.get_legend_handles_labels()

# 创建新的handles和labels，只包含要显示的变量名
new_handles = []
new_labels = []

for handle, var_name in zip(handles, plot_knn):
    if var_name in plot_knn:
        new_handles.append(handle)
        if var_name == 'knn_acc':
            new_labels.append('Acc')
        elif var_name == 'knn_mif':
            new_labels.append('Micro_F1')
        elif var_name == 'knn_maf':
            new_labels.append('Macro_F1')
        elif var_name == 'knn_mrc':
            new_labels.append('LR')
        elif var_name == 'knn_aca':
            new_labels.append('GM_Acc')
        elif var_name == 'knn_hm':
            new_labels.append('HM_Acc')

legend = ax.legend(new_handles, new_labels, bbox_to_anchor=(1.0, 1.0), loc='upper right')

ax.set_ylim(0, 1.3)
ax.set_xlabel('N_Components')
ax.set_ylabel('Value')
ax.set_title(r'$\mathbf{KNN}$')
# 启用网格线，设置虚线样式，不透明度为0.25
ax.grid(True, linestyle=':', alpha=0.50)
# 去掉四个边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# 显示图形
fig.savefig("./result_fig/knn_pca.png", dpi=600)
plt.close()




# SVM分类器
plot_svm = ['svm_acc', 'svm_mif', 'svm_maf', 'svm_mrc', 'svm_aca', 'svm_hm']
fig, ax = plt.subplots()
for i, var_name in enumerate(plot_svm):
    # 获取对应变量的值
    values = lists[var_name]

    # 计算误差带的上下界
    lower_bound = np.array(values) * 0.95
    upper_bound = np.array(values) * 1.05

    # 绘制误差带
    ax.fill_between(n_list_str, lower_bound, upper_bound, alpha=0.2, color=line_colors[i])

    # 绘制折线图，并设置线条颜色为line_colors中的对应颜色
    ax.plot(n_list_str, values, label=var_name, linestyle='-', color=line_colors[i])

    for x, y in zip(n_list_str, values):
        ax.annotate('{:.2f}'.format(y), (x, y), xytext=(0, 0), textcoords='offset points', ha='center', va='bottom', fontsize=6)

# 获取图例的handles和labels
handles, labels = ax.get_legend_handles_labels()

# 创建新的handles和labels，只包含要显示的变量名
new_handles = []
new_labels = []

for handle, var_name in zip(handles, plot_svm):
    if var_name in plot_svm:
        new_handles.append(handle)
        if var_name == 'svm_acc':
            new_labels.append('Acc')
        elif var_name == 'svm_mif':
            new_labels.append('Micro_F1')
        elif var_name == 'svm_maf':
            new_labels.append('Macro_F1')
        elif var_name == 'svm_mrc':
            new_labels.append('LR')
        elif var_name == 'svm_aca':
            new_labels.append('GM_Acc')
        elif var_name == 'svm_hm':
            new_labels.append('HM_Acc')


legend = ax.legend(new_handles, new_labels, bbox_to_anchor=(1.0, 1.0), loc='upper right')

ax.set_ylim(0, 1.1)
ax.set_xlabel('N_Components')
ax.set_ylabel('Value')
ax.set_title(r'$\mathbf{SVM}$')
# 启用网格线，设置虚线样式，不透明度为0.25
ax.grid(True, linestyle=':', alpha=0.50)
# 去掉四个边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# 显示图形
fig.savefig("./result_fig/svm_pca.png", dpi=600)
plt.close()




plot_qda = ['qda_acc', 'qda_mif', 'qda_maf', 'qda_mrc', 'qda_aca', 'qda_hm']
fig, ax = plt.subplots()
for i, var_name in enumerate(plot_qda):
    # 获取对应变量的值
    values = lists[var_name]

    # 计算误差带的上下界
    lower_bound = np.array(values) * 0.95
    upper_bound = np.array(values) * 1.05

    # 绘制误差带
    ax.fill_between(n_list_str, lower_bound, upper_bound, alpha=0.2, color=line_colors[i])

    # 绘制折线图，并设置线条颜色为line_colors中的对应颜色
    ax.plot(n_list_str, values, label=var_name, linestyle='-', color=line_colors[i])

    for x, y in zip(n_list_str, values):
        ax.annotate('{:.2f}'.format(y), (x, y), xytext=(0, 0), textcoords='offset points', ha='center', va='bottom', fontsize=6)

# 获取图例的handles和labels
handles, labels = ax.get_legend_handles_labels()

# 创建新的handles和labels，只包含要显示的变量名和对应的标签
new_handles = []
new_labels = []
for handle, var_name in zip(handles, plot_qda):
    if var_name in plot_qda:
        new_handles.append(handle)
        if var_name == 'qda_acc':
            new_labels.append('Acc')
        elif var_name == 'qda_mif':
            new_labels.append('Micro_F1')
        elif var_name == 'qda_maf':
            new_labels.append('Macro_F1')
        elif var_name == 'qda_mrc':
            new_labels.append('LR')
        elif var_name == 'qda_aca':
            new_labels.append('GM_Acc')
        elif var_name == 'qda_hm':
            new_labels.append('HM_Acc')

# 创建图例，并设置标签和位置
legend = ax.legend(new_handles, new_labels, bbox_to_anchor=(1.0, 1.0), loc='upper right')
ax.set_ylim(0, 1.4)
ax.set_xlabel('N_Components')
ax.set_ylabel('Value')
title_text = ax.set_title(r'$\mathbf{QDA}$')
# 设置每个字母的字体大小
# title_text.set_fontsize(10)
# 启用网格线，设置虚线样式，不透明度为0.25
ax.grid(True, linestyle=':', alpha=0.50)
# 去掉四个边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# 显示图形
fig.savefig("./result_fig/qda_pca.png", dpi=600)
plt.close()
