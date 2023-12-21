import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch, os, pathlib
import torch.nn.functional as F
from sklearn.decomposition import KernelPCA

save_dir = os.path.join(pathlib.Path(__file__).parent.absolute().__str__(), "Trained")

class confusion_matrix:
    def __init__(self, cls) -> None:
        self.cls = cls
        self.mat = np.zeros((cls, cls))
    def upd(self, y, y_hat):
        for row, col in zip(y, y_hat):
            self.mat[row][col] += 1
    def acc(self):
        tot, cor = 0, 0
        for i in range(self.cls):
            for j in range(self.cls):
                tot += self.mat[i][j]
                if i == j:
                    cor += self.mat[i][j]
        return cor / tot
    
# 两层隐层神经元
class Net(nn.Module):
    def __init__(self, w=32, decompose=100, cls=9):
        super(Net, self).__init__()
        self.f = torch.nn.Flatten() #
        self.dense1 = torch.nn.Linear(w * w * 3, decompose)
        self.dense2 = torch.nn.Linear(decompose, 9)
    def forward(self, x):
        x = self.f(x) #
        x = self.dense2(F.relu(self.dense1(x)))
        return x

class KNet(nn.Module):
    def __init__(self, w=32, decompose=100, cls=9) -> None:
        super(KNet, self).__init__()
        self.f = torch.nn.Flatten() #
        self.dense1 = torch.nn.Linear(w * w * 3, 256)
        self.dense2 = torch.nn.Linear(256, decompose)
        self.dense3 = nn.Linear(decompose, cls)
    def forward(self, x):
        x = self.f(x) #
        x = self.dense3(self.dense2(F.relu(self.dense1(x))))
        return x

# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet模型
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 构建ResNet-18模型
def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def train_epoch(net, train_iter, loss, updater, device):
    net.train()
    total_loss = 0
    num_batches = len(train_iter)

    co_mat = confusion_matrix(9)
    with tqdm(total=num_batches, desc='Epoch') as pbar:
        for i, (X, y) in enumerate(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            total_loss += l.mean().item()
            pbar.update(1)
            pbar.set_postfix({'Loss': total_loss / (i+1)})
            ### add metrics
            preds = torch.argmax(y_hat, dim=1)
            co_mat.upd(y, preds)
            ###
    print(f"acc:{co_mat.acc():.6f}")

def evaluate_model(net, data_iter, loss, device):
    net.eval()
    co_mat = confusion_matrix(9)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)# CrossEntropy 输入 logits,labels
            preds = torch.argmax(y_hat, dim=1)
            ## add metrics
            co_mat.upd(y, preds)
            ##
    print(f"acc:{co_mat.acc():.6f}")
    return co_mat

def save_model(net, epoch, acc):
    """epoch & acc : 准确率 """
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    torch.save(net.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_acc_{acc:.6f}.pt"))

def train(net, train_iter, test_iter, loss, num_epochs, updater, device):
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, updater, device)
        mat = evaluate_model(net, test_iter, loss, device)
        save_model(net, epoch + 1, mat.acc())

if __name__ == '__main__':
    model = ResNet18(9)
    # 打印模型结构
    print(model)
