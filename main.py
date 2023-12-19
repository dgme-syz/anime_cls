import os
from pathlib import Path
from datasets_.data_utils import DatasetFromFolder


if __name__ == '__main__':
    # 1. load 数据
    names = ["Blue Mountain", "Chino", "Chiya", "Cocoa", "Maya", \
         "Megumi", "Mocha", "Rize", "Sharo"]
    base_dir = Path(__file__).parent.absolute().__str__()
    data_dir = os.path.join(base_dir, "datasets_", "data")
    test_data = DatasetFromFolder(os.path.join(data_dir, "ANIME"), names)
    train_data = DatasetFromFolder(os.path.join(data_dir, "DANBOORU"), names)










