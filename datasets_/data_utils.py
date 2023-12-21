import cv2, torch
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_pil_image

def preprocess(img):
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),  # 转换为Tensor格式
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    return transform(img)

class DatasetFromFolder(Dataset):
    def __init__(self, dir_, cls):
        super(DatasetFromFolder, self).__init__()
        self.f = nn.Flatten()
        self.img_filenames = []
        for x in range(len(cls)):
            for img in listdir(join(dir_, cls[x])):
                self.img_filenames.append((join(dir_, cls[x], img), x))
    def ml_data(self):
        X, y = [], []
        for file_path, label in self.img_filenames:
            # bgr -> rgb
            img = preprocess(Image.open(file_path))
            X.append(self.f(img.unsqueeze(0))[0].detach().numpy()) # 与神经网络一致
            y.append(label)
        return np.array(X), y
    def __getitem__(self, index):
        img, label = self.img_filenames[index]
        return preprocess(Image.open(img)), label 
    def __len__(self):
        return len(self.img_filenames)

if __name__ == '__main__':
    ar = np.arange(12).reshape((2, 2, 3))
    print(ar)
    print(preprocess(ar))
    f = nn.Flatten()
    z = f(preprocess(ar).unsqueeze(0))
    print(z.view(ar.shape[2], ar.shape[0], ar.shape[1]).permute((1, 2, 0)))# 重要




