import cv2
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_pil_image

names = ["Blue Mountain", "Chino", "Chiya", "Cocoa", "Maya", \
         "Megumi", "Mocha", "Rize", "Sharo"]

def preprocess(img):
    """
    读取 ndarray 形式的图片，并 resize 为 64 x 64 x 3
    """
    if isinstance(img, np.array):
        img = to_pil_image(img)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    return transform(img)

class DatasetFromFolder(Dataset):
    def __init__(self, dir_):
        super(DatasetFromFolder, self).__init__()
        names = []
        self.img_filenames = []
        for x in range(9):
            for img in listdir(join(dir_, names[x])):
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
        return preprocess(Image.open(img)), label 
    def __len__(self):
        return len(self.img_filenames)





