import cv2, torch
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_pil_image

def preprocess(img):
    """
    读取 ndarray 形式的图片，并 resize 为 64 x 64 x 3
    """
    if isinstance(img, np.array):
        img = to_pil_image(img)
    return img

class DatasetFromFolder(Dataset):
    def __init__(self, dir_, cls):
        super(DatasetFromFolder, self).__init__()
        self.img_filenames = []
        for x in range(len(cls)):
            for img in listdir(join(dir_, cls[x])):
                self.img_filenames.append((join(dir_, str(x), img), x))
    def ml_data(self):
        X, y = [], []
        for file_path, label in self.img_filenames:
            # bgr -> rgb
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label)
        return X, y
    def __getitem__(self, index):
        img, label = self.img_filenames[index]
        return transforms.ToTensor(preprocess(Image.open(img))), label 
    def __len__(self):
        return len(self.img_filenames)





