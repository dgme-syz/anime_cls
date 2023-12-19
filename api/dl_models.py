import torch.nn as nn
import torch, tqdm, os, pathlib
import torch.nn.functional as F

save_dir = os.path.join(pathlib.Path(__file__).parent.absolute().__str__(), "Trained")

class confusion_matrix:
    def __init__(self, cls) -> None:
        self.cls = cls
        self.mat = [[] for _ in range(cls)]
    def upd(self, y, y_hat):
        for row, col in zip(y, y_hat):
            self.mat[row, col] += 1
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
    def __init__(self, w = 64, decompose=100):
        super(Net, self).__init__()
        self.dense1 = torch.nn.Linear(w * w * 3, decompose)
        self.dense2 = torch.nn.Linear(decompose, 9)
    def forward(self, x):
        x = torch.flatten(x)
        x = self.dense2(F.tanh(self.dense1(x)))

def train_epoch(net, train_iter, loss, updater):
    net.train()
    total_loss = 0
    num_batches = len(train_iter)

    co_mat = confusion_matrix(9)
    with tqdm(total=num_batches, desc='Epoch') as pbar:
        for i, (X, y) in enumerate(train_iter):
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
    print(f"acc:{co_mat.acc:.6f}")

def evaluate_model(net, data_iter, loss):
    net.eval()
    co_mat = confusion_matrix(9)
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            l = loss(y_hat, y)# CrossEntropy 输入 logits,labels
            preds = torch.argmax(y_hat, dim=1)
            ## add metrics
            co_mat.upd(y, preds)
            ##
    return co_mat

def save_model(net, epoch, acc):
    """epoch & acc : 准确率 """
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    torch.save(net.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_acc_{acc:.6f}.pt"))

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, updater)
        mat = evaluate_model(net, test_iter, loss)
        if (epoch + 1) % 5 == 0:
            save_model(net, epoch + 1, mat.acc)

if __name__ == '__main__':
    model = Net()
    # 打印模型结构
    print(model)
