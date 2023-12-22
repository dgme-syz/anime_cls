import os, torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datasets_.data_utils import DatasetFromFolder
from torch.utils.data import DataLoader
from api.dl_models import Net, train, ResNet18, KNet
from api.ml_models import *
import torch.nn.functional as F
from datasets_.data_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
names = ["Blue Mountain", "Chino", "Chiya", "Cocoa", "Maya", \
        "Megumi", "Mocha", "Rize", "Sharo"]
base_dir = Path(__file__).parent.absolute().__str__()
data_dir = os.path.join(base_dir, "datasets_", "data")


def dl():
    # 0. print 设备
    print(device)

    # 1. load 数据
    train_data = DatasetFromFolder(os.path.join(data_dir, "ANIME"), names)
    test_data = DatasetFromFolder(os.path.join(data_dir, "DANBOORU"), names)

    train_iter = DataLoader(train_data, batch_size=256, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=256, shuffle=False)

    # 2. load 模型
    net = ResNet18(9).to(device)
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(lr=0.001, params = net.parameters())
    num_epochs = 10

    # 3. train 模型
    train(net, train_iter, test_iter, loss, num_epochs, optimizer, device)


def ml():
    # 1. load 数据
    train_data = DatasetFromFolder(os.path.join(data_dir, "ANIME"), names)
    test_data = DatasetFromFolder(os.path.join(data_dir, "DANBOORU"), names)

    tr_X, tr_y = train_data.ml_data()
    te_X, te_y = test_data.ml_data()

    # 神经网络需要注释
    
    scaler = StandardScaler()

    tr_X = scaler.fit_transform(tr_X)
    te_X = scaler.transform(te_X)



    # 使用你的 ml_model 进行测试
    print(f"训练集尺寸: {np.array(tr_X).shape} 测试集尺寸: {np.array(te_X).shape}")
    tr_X = np.array(tr_X)
    te_X = np.array(te_X)

    ###
    #  测试

    def pca(tr_X, tr_y, te_X, te_y):
        n_list = [10,50,100,200,500]
        n_list_str = ['10', '50', '100', '200', '500']
        variance_ratio = []
        # 定义列表名
        list_names = ['dtree_acc', 'dtree_time', 'dtree_mif', 'dtree_maf', 'dtree_mrc', 'dtree_aca', 'dtree_hm',
                      'mlr_acc', 'mlr_time', 'mlr_mif', 'mlr_maf', 'mlr_mrc', 'mlr_aca', 'mlr_hm',
                      'knn_acc', 'knn_time', 'knn_mif', 'knn_maf', 'knn_mrc', 'knn_aca', 'knn_hm',
                      'svm_acc', 'svm_time', 'svm_mif', 'svm_maf', 'svm_mrc', 'svm_aca', 'svm_hm',
                      'qda_acc', 'qda_time', 'qda_mif', 'qda_maf', 'qda_mrc', 'qda_aca', 'qda_hm']
        # 创建空列表的字典
        lists = {name: [] for name in list_names}
        for i in n_list:
            train_X, test_X, vr = pca_method(tr_X, te_X, i)
            variance_ratio.append(vr)

            acc, time, mif, maf, mrc, aca, hm = decision_tree_method(train_X, tr_y, test_X, te_y)
            lists['dtree_acc'].append(acc)
            lists['dtree_time'].append(time)
            lists['dtree_mif'].append(mif)
            lists['dtree_maf'].append(maf)
            lists['dtree_mrc'].append(mrc)
            lists['dtree_aca'].append(aca)
            lists['dtree_hm'].append(hm)

            acc, time, mif, maf, mrc, aca, hm = multivariables_linear_regression(train_X, tr_y, test_X, te_y)
            lists['mlr_acc'].append(acc)
            lists['mlr_time'].append(time)
            lists['mlr_mif'].append(mif)
            lists['mlr_maf'].append(maf)
            lists['mlr_mrc'].append(mrc)
            lists['mlr_aca'].append(aca)
            lists['mlr_hm'].append(hm)

            acc, time, mif, maf, mrc, aca, hm = knn_method(train_X, tr_y, test_X, te_y)
            lists['knn_acc'].append(acc)
            lists['knn_time'].append(time)
            lists['knn_mif'].append(mif)
            lists['knn_maf'].append(maf)
            lists['knn_mrc'].append(mrc)
            lists['knn_aca'].append(aca)
            lists['knn_hm'].append(hm)

            acc, time, mif, maf, mrc, aca, hm = svm_method(train_X, tr_y, test_X, te_y)
            lists['svm_acc'].append(acc)
            lists['svm_time'].append(time)
            lists['svm_mif'].append(mif)
            lists['svm_maf'].append(maf)
            lists['svm_mrc'].append(mrc)
            lists['svm_aca'].append(aca)
            lists['svm_hm'].append(hm)

            acc, time, mif, maf, mrc, aca, hm = qda_method(train_X, tr_y, test_X, te_y)
            lists['qda_acc'].append(acc)
            lists['qda_time'].append(time)
            lists['qda_mif'].append(mif)
            lists['qda_maf'].append(maf)
            lists['qda_mrc'].append(mrc)
            lists['qda_aca'].append(aca)
            lists['qda_hm'].append(hm)

        print("\n")
        output_format = "在10, 50, 100, 200, 500维度下，决策树的准确率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['dtree_acc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，决策树的时间分别为: {}"
        output = output_format.format(', '.join(map(str, lists['dtree_time'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，决策树的micro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['dtree_mif'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，决策树的macro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['dtree_maf'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，决策树的最小召回率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['dtree_mrc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，决策树的类准确率平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['dtree_aca'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，决策树的类准确率调和平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['dtree_hm'])))
        print(output)
        print("\n")

        output_format = "在10, 50, 100, 200, 500维度下，多因变量线性回归的准确率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['mlr_acc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，多因变量线性回归的时间分别为: {}"
        output = output_format.format(', '.join(map(str, lists['mlr_time'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，多因变量线性回归的micro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['mlr_mif'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，多因变量线性回归的macro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['mlr_maf'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，多因变量线性回归的最小召回率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['mlr_mrc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，多因变量线性回归的类准确率平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['mlr_aca'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，多因变量线性回归的类准确率调和平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['mlr_hm'])))
        print(output)
        print("\n")

        output_format = "在10, 50, 100, 200, 500维度下，KNN的准确率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['knn_acc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，KNN的时间分别为: {}"
        output = output_format.format(', '.join(map(str, lists['knn_time'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，KNN的micro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['knn_mif'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，KNN的macro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['knn_maf'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，KNN的最小召回率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['knn_mrc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，KNN的类准确率平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['knn_aca'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，KNN的类准确率调和平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['knn_hm'])))
        print(output)
        print("\n")

        output_format = "在10, 50, 100, 200, 500维度下，SVM的准确率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['svm_acc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，SVM的时间分别为: {}"
        output = output_format.format(', '.join(map(str, lists['svm_time'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，SVM的micro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['svm_mif'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，SVM的macro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['svm_maf'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，SVM的最小召回率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['svm_mrc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，SVM的类准确率平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['svm_aca'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，SVM的类准确率调和平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['svm_hm'])))
        print(output)
        print("\n")

        output_format = "在10, 50, 100, 200, 500维度下，QDA的准确率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['qda_acc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，QDA的时间分别为: {}"
        output = output_format.format(', '.join(map(str, lists['qda_time'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，QDA的micro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['qda_mif'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，QDA的macro_F1分别为: {}"
        output = output_format.format(', '.join(map(str, lists['qda_maf'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，QDA的最小召回率分别为: {}"
        output = output_format.format(', '.join(map(str, lists['qda_mrc'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，QDA的类准确率平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['qda_aca'])))
        print(output)
        output_format = "在10, 50, 100, 200, 500维度下，QDA的类准确率调和平均值分别为: {}"
        output = output_format.format(', '.join(map(str, lists['qda_hm'])))
        print(output)
        print("\n")

        # 绘制PCA方差解释率随维度的变化
        lower_curve = [x - 0.05 for x in variance_ratio]
        upper_curve = [x + 0.05 for x in variance_ratio]
        fig, ax = plt.subplots()
        ax.plot(n_list_str, variance_ratio, color='#144a74')
        ax.fill_between(n_list_str, lower_curve, upper_curve, alpha=0.35, color='#93b5cf')

        ax.set(xlabel='降维维度', ylabel='方差解释率',
               title='PCA方差解释率随维度的变化')
        lower_bound = 0.5
        upper_bound = 1.0
        ax.set_ylim(lower_bound, upper_bound)
        ax.set_yticks([lower_bound + i*(upper_bound-lower_bound)/5 for i in range(6)])
        ax.grid(True, color = '#5e7987', linestyle=':', alpha=0.20)
        for i, j in zip(n_list_str, variance_ratio):
            ax.text(i, j, str(round(j, 2)), ha='center', va='bottom', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        fig.savefig("./result_fig/variance_ratio.png", dpi=600)
        plt.close()
        '''
        plot_dtree = ['dtree_acc', 'dtree_mif', 'dtree_maf', 'dtree_mrc', 'dtree_aca', 'dtree_hm']
        line_colors = ['#dad8e5', '#b4cfe2', '#a0c1d4', '#91a3bb', '#536d8e', '#2177b8']
        '''

    def flda(tr_X, tr_y, te_X, te_y):

        train_X, test_X = flda_method(tr_X, tr_y, te_X)

        # 定义变量名列表
        variable_names = ['dtree_acc', 'dtree_time', 'dtree_mif', 'dtree_maf', 'dtree_mrc', 'dtree_aca', 'dtree_hm',
                          'mlr_acc', 'mlr_time', 'mlr_mif', 'mlr_maf', 'mlr_mrc', 'mlr_aca', 'mlr_hm',
                          'knn_acc', 'knn_time', 'knn_mif', 'knn_maf', 'knn_mrc', 'knn_aca', 'knn_hm',
                          'svm_acc', 'svm_time', 'svm_mif', 'svm_maf', 'svm_mrc', 'svm_aca', 'svm_hm',
                          'qda_acc', 'qda_time', 'qda_mif', 'qda_maf', 'qda_mrc', 'qda_aca', 'qda_hm']
        # 创建空字典
        variables = {}
        # 使用循环批量创建变量，并赋值为 0
        for name in variable_names:
            variables[name] = 0

        # step1 再次检查多重共线性
        # check_cov(train_X)

        # 如下填写各个模型的测试信息
        variables['dtree_acc'], variables['dtree_time'], variables['dtree_mif'], variables['dtree_maf'], variables['dtree_mrc'], \
        variables['dtree_aca'], variables['dtree_hm'] = decision_tree_method(train_X, tr_y, test_X, te_y)

        variables['mlr_acc'], variables['mlr_time'], variables['mlr_mif'], variables['mlr_maf'], variables['mlr_mrc'], \
        variables['mlr_aca'], variables['mlr_hm'] = multivariables_linear_regression(train_X, tr_y, test_X, te_y)

        variables['knn_acc'], variables['knn_time'], variables['knn_mif'], variables['knn_maf'], variables['knn_mrc'], \
        variables['knn_aca'], variables['knn_hm'] = knn_method(train_X, tr_y, test_X, te_y)

        variables['svm_acc'], variables['svm_time'], variables['svm_mif'], variables['svm_maf'], variables['svm_mrc'], \
        variables['svm_aca'], variables['svm_hm'] = svm_method(train_X, tr_y, test_X, te_y)

        variables['qda_acc'], variables['qda_time'], variables['qda_mif'], variables['qda_maf'], variables['qda_mrc'], \
        variables['qda_aca'], variables['qda_hm'] = qda_method(train_X, tr_y, test_X, te_y)
        ####

        print("\n")
        output = "在flda降维至{}维下，决策树的准确率为:{}".format(8, variables['dtree_acc'])
        print(output)
        output = "在flda降维至{}维下，决策树的时间为:{}".format(8, variables['dtree_time'])
        print(output)
        output = "在flda降维至{}维下，决策树的micro_F1为:{}".format(8, variables['dtree_mif'])
        print(output)
        output = "在flda降维至{}维下，决策树的macro_F1为:{}".format(8, variables['dtree_maf'])
        print(output)
        output = "在flda降维至{}维下，决策树的最小召回率为:{}".format(8, variables['dtree_mrc'])
        print(output)
        output = "在flda降维至{}维下，决策树的类准确率平均值为:{}".format(8, variables['dtree_aca'])
        print(output)
        output = "在flda降维至{}维下，决策树的类准确率调和平均值为:{}".format(8, variables['dtree_hm'])
        print(output)
        print("\n")

        output = "在flda降维至{}维下，多因变量线性回归的准确率为:{}".format(8, variables['mlr_acc'])
        print(output)
        output = "在flda降维至{}维下，多因变量线性回归的时间为:{}".format(8, variables['mlr_time'])
        print(output)
        output = "在flda降维至{}维下，多因变量线性回归的micro_F1为:{}".format(8, variables['mlr_mif'])
        print(output)
        output = "在flda降维至{}维下，多因变量线性回归的macro_F1为:{}".format(8, variables['mlr_maf'])
        print(output)
        output = "在flda降维至{}维下，多因变量线性回归的最小召回率为:{}".format(8, variables['mlr_mrc'])
        print(output)
        output = "在flda降维至{}维下，多因变量线性回归的类准确率平均值为:{}".format(8, variables['mlr_aca'])
        print(output)
        output = "在flda降维至{}维下，多因变量线性回归的类准确率调和平均值为:{}".format(8, variables['mlr_hm'])
        print(output)
        print("\n")

        output = "在flda降维至{}维下，KNN的准确率为:{}".format(8, variables['knn_acc'])
        print(output)
        output = "在flda降维至{}维下，KNN的时间为:{}".format(8, variables['knn_time'])
        print(output)
        output = "在flda降维至{}维下，KNN的micro_F1为:{}".format(8, variables['knn_mif'])
        print(output)
        output = "在flda降维至{}维下，KNN的macro_F1为:{}".format(8, variables['knn_maf'])
        print(output)
        output = "在flda降维至{}维下，KNN的最小召回率为:{}".format(8, variables['knn_mrc'])
        print(output)
        output = "在flda降维至{}维下，KNN的类准确率平均值为:{}".format(8, variables['knn_aca'])
        print(output)
        output = "在flda降维至{}维下，KNN的类准确率调和平均值为:{}".format(8, variables['knn_hm'])
        print(output)
        print("\n")

        output = "在flda降维至{}维下，SVM的准确率为:{}".format(8, variables['svm_acc'])
        print(output)
        output = "在flda降维至{}维下，SVM的时间为:{}".format(8, variables['svm_time'])
        print(output)
        output = "在flda降维至{}维下，SVM的micro_F1为:{}".format(8, variables['svm_mif'])
        print(output)
        output = "在flda降维至{}维下，SVM的macro_F1为:{}".format(8, variables['svm_maf'])
        print(output)
        output = "在flda降维至{}维下，SVM的最小召回率为:{}".format(8, variables['svm_mrc'])
        print(output)
        output = "在flda降维至{}维下，SVM的类准确率平均值为:{}".format(8, variables['svm_aca'])
        print(output)
        output = "在flda降维至{}维下，SVM的类准确率调和平均值为:{}".format(8, variables['svm_hm'])
        print(output)
        print("\n")

        output = "在flda降维至{}维下，QDA的准确率为:{}".format(8, variables['qda_acc'])
        print(output)
        output = "在flda降维至{}维下，QDA的时间为:{}".format(8, variables['qda_time'])
        print(output)
        output = "在flda降维至{}维下，QDA的micro_F1为:{}".format(8, variables['qda_mif'])
        print(output)
        output = "在flda降维至{}维下，QDA的macro_F1为:{}".format(8, variables['qda_maf'])
        print(output)
        output = "在flda降维至{}维下，QDA的最小召回率为:{}".format(8, variables['qda_mrc'])
        print(output)
        output = "在flda降维至{}维下，QDA的类准确率平均值为:{}".format(8, variables['qda_aca'])
        print(output)
        output = "在flda降维至{}维下，QDA的类准确率调和平均值为:{}".format(8, variables['qda_hm'])
        print(output)

    def dl_decompose1(tr_X, tr_y, te_X, te_y):
        net = Net()
        net.load_state_dict(torch.load(os.path.join(base_dir, "api", "Trained", "base.pt"), \
                                       map_location=device))

        def convert(x):
            w, h = x.shape
            new_x = []
            for i in range(w):
                new_x.append(x[i].reshape((3, 32, 32)))
            new_x = torch.tensor(np.array(new_x), dtype=torch.float32)
            print(new_x.shape)
            return F.relu(net.dense1(net.f(new_x))).detach().numpy()

        train_X, test_X = convert(tr_X), convert(te_X)
        assert train_X.shape[1] == 100

        # 定义变量名列表
        variable_names = ['dtree_acc', 'dtree_time', 'dtree_mif', 'dtree_maf', 'dtree_mrc', 'dtree_aca', 'dtree_hm',
                          'mlr_acc', 'mlr_time', 'mlr_mif', 'mlr_maf', 'mlr_mrc', 'mlr_aca', 'mlr_hm',
                          'knn_acc', 'knn_time', 'knn_mif', 'knn_maf', 'knn_mrc', 'knn_aca', 'knn_hm',
                          'svm_acc', 'svm_time', 'svm_mif', 'svm_maf', 'svm_mrc', 'svm_aca', 'svm_hm',
                          'qda_acc', 'qda_time', 'qda_mif', 'qda_maf', 'qda_mrc', 'qda_aca', 'qda_hm']
        # 创建空字典
        variables = {}
        # 使用循环批量创建变量，并赋值为 0
        for name in variable_names:
            variables[name] = 0

        # step1 再次检查多重共线性
        # check_cov(train_X)

        # 如下填写各个模型的测试信息
        variables['dtree_acc'], variables['dtree_time'], variables['dtree_mif'], variables['dtree_maf'], variables['dtree_mrc'], \
        variables['dtree_aca'], variables['dtree_hm'] = decision_tree_method(train_X, tr_y, test_X, te_y)

        variables['mlr_acc'], variables['mlr_time'], variables['mlr_mif'], variables['mlr_maf'], variables['mlr_mrc'], \
        variables['mlr_aca'], variables['mlr_hm'] = multivariables_linear_regression(train_X, tr_y, test_X, te_y)

        variables['knn_acc'], variables['knn_time'], variables['knn_mif'], variables['knn_maf'], variables['knn_mrc'], \
        variables['knn_aca'], variables['knn_hm'] = knn_method(train_X, tr_y, test_X, te_y)

        variables['svm_acc'], variables['svm_time'], variables['svm_mif'], variables['svm_maf'], variables['svm_mrc'], \
        variables['svm_aca'], variables['svm_hm'] = svm_method(train_X, tr_y, test_X, te_y)

        variables['qda_acc'], variables['qda_time'], variables['qda_mif'], variables['qda_maf'], variables['qda_mrc'], \
        variables['qda_aca'], variables['qda_hm'] = qda_method(train_X, tr_y, test_X, te_y)
        ####

        print("\n")
        output = "在神经网络降维至{}维下，决策树的准确率为:{}".format(100, variables['dtree_acc'])
        print(output)
        output = "在神经网络降维至{}维下，决策树的时间为:{}".format(100, variables['dtree_time'])
        print(output)
        output = "在神经网络降维至{}维下，决策树的micro_F1为:{}".format(100, variables['dtree_mif'])
        print(output)
        output = "在神经网络降维至{}维下，决策树的macro_F1为:{}".format(100, variables['dtree_maf'])
        print(output)
        output = "在神经网络降维至{}维下，决策树的最小召回率为:{}".format(100, variables['dtree_mrc'])
        print(output)
        output = "在神经网络降维至{}维下，决策树的类准确率平均值为:{}".format(100, variables['dtree_aca'])
        print(output)
        output = "在神经网络降维至{}维下，决策树的类准确率调和平均值为:{}".format(100, variables['dtree_hm'])
        print(output)
        print("\n")

        output = "在神经网络降维至{}维下，多因变量线性回归的准确率为:{}".format(100, variables['mlr_acc'])
        print(output)
        output = "在神经网络降维至{}维下，多因变量线性回归的时间为:{}".format(100, variables['mlr_time'])
        print(output)
        output = "在神经网络降维至{}维下，多因变量线性回归的micro_F1为:{}".format(100, variables['mlr_mif'])
        print(output)
        output = "在神经网络降维至{}维下，多因变量线性回归的macro_F1为:{}".format(100, variables['mlr_maf'])
        print(output)
        output = "在神经网络降维至{}维下，多因变量线性回归的最小召回率为:{}".format(100, variables['mlr_mrc'])
        print(output)
        output = "在神经网络降维至{}维下，多因变量线性回归的类准确率平均值为:{}".format(100, variables['mlr_aca'])
        print(output)
        output = "在神经网络降维至{}维下，多因变量线性回归的类准确率调和平均值为:{}".format(100, variables['mlr_hm'])
        print(output)
        print("\n")

        output = "在神经网络降维至{}维下，KNN的准确率为:{}".format(100, variables['knn_acc'])
        print(output)
        output = "在神经网络降维至{}维下，KNN的时间为:{}".format(100, variables['knn_time'])
        print(output)
        output = "在神经网络降维至{}维下，KNN的micro_F1为:{}".format(100, variables['knn_mif'])
        print(output)
        output = "在神经网络降维至{}维下，KNN的macro_F1为:{}".format(100, variables['knn_maf'])
        print(output)
        output = "在神经网络降维至{}维下，KNN的最小召回率为:{}".format(100, variables['knn_mrc'])
        print(output)
        output = "在神经网络降维至{}维下，KNN的类准确率平均值为:{}".format(100, variables['knn_aca'])
        print(output)
        output = "在神经网络降维至{}维下，KNN的类准确率调和平均值为:{}".format(100, variables['knn_hm'])
        print(output)
        print("\n")

        output = "在神经网络降维至{}维下，SVM的准确率为:{}".format(100, variables['svm_acc'])
        print(output)
        output = "在神经网络降维至{}维下，SVM的时间为:{}".format(100, variables['svm_time'])
        print(output)
        output = "在神经网络降维至{}维下，SVM的micro_F1为:{}".format(100, variables['svm_mif'])
        print(output)
        output = "在神经网络降维至{}维下，SVM的macro_F1为:{}".format(100, variables['svm_maf'])
        print(output)
        output = "在神经网络降维至{}维下，SVM的最小召回率为:{}".format(100, variables['svm_mrc'])
        print(output)
        output = "在神经网络降维至{}维下，SVM的类准确率平均值为:{}".format(100, variables['svm_aca'])
        print(output)
        output = "在神经网络降维至{}维下，SVM的类准确率调和平均值为:{}".format(100, variables['svm_hm'])
        print(output)
        print("\n")

        output = "在神经网络降维至{}维下，QDA的准确率为:{}".format(100, variables['qda_acc'])
        print(output)
        output = "在神经网络降维至{}维下，QDA的时间为:{}".format(100, variables['qda_time'])
        print(output)
        output = "在神经网络降维至{}维下，QDA的micro_F1为:{}".format(100, variables['qda_mif'])
        print(output)
        output = "在神经网络降维至{}维下，QDA的macro_F1为:{}".format(100, variables['qda_maf'])
        print(output)
        output = "在神经网络降维至{}维下，QDA的最小召回率为:{}".format(100, variables['qda_mrc'])
        print(output)
        output = "在神经网络降维至{}维下，QDA的类准确率平均值为:{}".format(100, variables['qda_aca'])
        print(output)
        output = "在神经网络降维至{}维下，QDA的类准确率调和平均值为:{}".format(100, variables['qda_hm'])
        print(output)



    pca(tr_X, tr_y, te_X, te_y)
    # flda(tr_X, tr_y, te_X, te_y)
    # dl_decompose1(tr_X, tr_y, te_X, te_y)


    # train_X, test_X = flda_method(tr_X, tr_y, te_X)
    # acc, time, mif, maf, mrc, aca, hm = decision_tree_method(train_X, tr_y, test_X, te_y)
    # print(mrc, aca, hm)
    ###


if __name__ == '__main__':
    # dl()
    ml()