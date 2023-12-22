import os, torch
import time

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
from pylab import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
names = ["Blue Mountain", "Chino", "Chiya", "Cocoa", "Maya", \
        "Megumi", "Mocha", "Rize", "Sharo"]
base_dir = Path(__file__).parent.absolute().__str__()
data_dir = os.path.join(base_dir, "datasets_", "data")

mpl.rcParams['font.sans-serif'] = ['SimHei']

def decompose_visulaize(train_X, train_y):
        net = Net(decompose=2)
        net.load_state_dict(torch.load(os.path.join(base_dir, "api", "Trained", "decompose=2.pt"), \
                                       map_location=device))
        def convert(x):
            w, h = x.shape
            new_x = []
            for i in range(w):
                new_x.append(x[i].reshape((3, 32, 32)))
            new_x = torch.tensor(np.array(new_x), dtype=torch.float32)
            return net.dense1(net.f(new_x)).detach().numpy()

        nn_X = convert(train_X)
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        pca_X, _, _ = pca_method(train_X, np.zeros((1, train_X.shape[1])), 2)
        flda_X, _ = flda_method(train_X, train_y, np.zeros((1, train_X.shape[1])), 2)
        # TwoScatter(pca_X, train_y, "pca1")
        # TwoScatter(flda_X, train_y, "flda1")
        TwoScatter(nn_X, train_y, "nn1")

def plot_cls(tr_y, te_y):
    tr_counts = np.bincount(tr_y)
    te_counts = np.bincount(te_y)

    # 设置颜色
    tr_color = '#619ac3'
    te_color = '#cbe99d'

    # 绘制水平的条形图
    fig, ax = plt.subplots()
    bar_height = 0.35  # 设置条形高度
    # 绘制训练集条形
    bars1 = ax.barh(np.arange(len(tr_counts)), tr_counts, bar_height, label='训练集', color=tr_color)
    # 绘制测试集条形
    bars2 = ax.barh(np.arange(len(te_counts)) + bar_height, te_counts, bar_height, label='测试集', color=te_color)
    # 添加标签、标题和图例
    ax.set_ylabel('类别')
    ax.set_xlabel('样本数目')
    ax.set_title('训练集和测试集样本数目分布')
    ax.set_yticks(np.arange(len(tr_counts)) + bar_height / 2)
    ax.set_yticklabels(np.arange(len(tr_counts)))
    # 移除右边和上边的坐标轴线和刻度
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(right=False, top=False)
    ax.legend()
    plt.grid(True, linestyle=':', alpha = 0.35)
    plt.savefig('./img/o.png', dpi=600)
    plt.show()

def dl():
    # 0. print 设备
    print(device)

    # 1. load 数据
    train_data = DatasetFromFolder(os.path.join(data_dir, "ANIME"), names)
    test_data = DatasetFromFolder(os.path.join(data_dir, "DANBOORU"), names)

    train_iter = DataLoader(train_data, batch_size=256, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=256, shuffle=False)

    # 2. load 模型
    net = Net(decompose=2).to(device)
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(lr=0.001, params = net.parameters())
    num_epochs = 10

    # 3. train 模型
    train(net, train_iter, test_iter, loss, num_epochs, optimizer, device)


def ml():
    # 开幕雷击
    print("第一次加载（读入数据）比较缓慢，请耐心等待。")

    # 1. load 数据
    train_data = DatasetFromFolder(os.path.join(data_dir, "ANIME"), names)
    test_data = DatasetFromFolder(os.path.join(data_dir, "DANBOORU"), names)

    tr_X, tr_y = train_data.ml_data()
    te_X, te_y = test_data.ml_data()
    
    # 2. 绘制散点图, 数据分布
    # decompose_visulaize(tr_X, tr_y)
    print(f"训练集尺寸: {np.array(tr_X).shape} 测试集尺寸: {np.array(te_X).shape}")
    # plot_cls(tr_y, tr_y)

    # 3. 【降维】 后可以使用检验正态性
    # check_nor(tr_X, tr_y)

    
    
    # 4. 打印训练信息
    # dl_decompose1(tr_X, tr_y, te_X, te_y)

    scaler = StandardScaler()
    tr_X = scaler.fit_transform(tr_X)
    te_X = scaler.transform(te_X)
    # pca(tr_X, tr_y, te_X, te_y)
    flda(tr_X, tr_y, te_X, te_y)

if __name__ == '__main__':
    # dl()
    ml()






