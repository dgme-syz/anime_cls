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
from scipy.stats import chi2
import matplotlib.pyplot as plt

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
    net = Net().to(device)
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

    scaler = StandardScaler()

    tr_X = scaler.fit_transform(tr_X)
    te_X = scaler.transform(te_X)

    tr_X_array = np.array(tr_X)
    te_X_array = np.array(te_X)
    # 使用你的 ml_model 进行测试
    print(f"训练集尺寸: {np.array(tr_X).shape} 测试集尺寸: {np.array(te_X).shape}")

    #  测试
    def pca(train_X, train_y, test_X, test_y):
        train_X, test_X = pca_method(train_X, test_X)
        check_nor(train_X, train_y)

        #### 如下填写各个模型的测试信息

        ####
    # pca(tr_X, tr_y, te_X, te_y)
    def dl_decompose(train_X, train_y, test_X, test_y):
        net = Net()
        net.load_state_dict(torch.load(os.path.join(base_dir, "api", "Trained", "base.pt")))
        def convert(data_):
            return net.dense1(torch.tensor(data_, dtype=torch.float32).unsqueeze(0))[0].detach().numpy()
        train_X, test_X = convert(train_X), convert(test_X)
        assert train_X.shape[1] == 100
        check_nor(train_X, train_y)
    # dl_decompose(tr_X, tr_y, te_X, te_y)

    def flda(train_X, train_y, test_X, test_y):
        train_X, test_X = flda_method(train_X, test_X)
    # flda(tr_X, tr_y, te_X, te_y)
    ###


if __name__ == '__main__':
    # dl()
    ml()








