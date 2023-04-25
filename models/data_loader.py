import os
import torch.utils.data as data
import torchvision.transforms as T
import json
from torchvision.io import read_image
import matplotlib.pyplot as plt

class HE_Dataset(data.Dataset):
    
    def __init__(self, data_dir, labels_file) -> None:
        super().__init__()
        self.data_dir = data_dir
        f = open(os.path.join(data_dir, labels_file), 'r')
        self.labels = json.load(f)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image =  read_image(os.path.join(self.data_dir, self.labels[str(index)][0]))
        image = image / 255
        label = self.labels[str(index)][1]
        return image, label
        
if __name__ == '__main__':
    dataset = HE_Dataset('data/pcam/train', 'labels.json')
    # dataset = HE_Dataset('data/pcam/test', 'labels.json)
    # dataset = HE_Dataset('data/pcam/valid', 'labels.json)
    x, y = dataset.__getitem__(0)
    print(dataset.__len__())
    print(x, y)
    print(x.shape, y)
    print(x.dtype, y)
    plt.imshow(x.permute(1, 2, 0))
    plt.show()