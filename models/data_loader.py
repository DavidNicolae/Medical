import os
import h5py
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import json
from torchvision.io import read_image
from PIL import Image
import matplotlib.pyplot as plt

DIR = 'h5_data'
FILE_FORMAT = 'camelyonpatch_level_2_split_{}_{}.h5'

def h5_to_jpeg(path):
    try:
        os.makedirs(path)
    except:
        print('Data already prepared')
        return
    
    for stage in ['train', 'test', 'valid']:
        inner_path = os.path.join(path, stage)
        os.makedirs(inner_path)
        
        images = h5py.File(os.path.join(DIR, FILE_FORMAT.format(stage, 'x')), 'r')
        labels = h5py.File(os.path.join(DIR, FILE_FORMAT.format(stage, 'y')), 'r')
        x_key = list(images.keys())[0]
        y_key = list(labels.keys())[0]
        
        for index, image in enumerate(images[x_key]):
            img = Image.fromarray(image)
            img.save(os.path.join(inner_path, 'img_' + str(index) + '.jpeg'), 'jpeg')
        
        labels = {'labels': labels[y_key][:].flatten().tolist()}
        with open(os.path.join(inner_path, 'labels.json'), 'w') as f:
            json.dump(labels, f, indent=4)

class HE_Dataset(data.Dataset):
    
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        files = os.listdir(data)
        self.images = files[:-1]
        labels_file = open(os.path.join(data, files[-1]), 'r')
        self.labels = json.load(labels_file)['labels']
        
        self.transform = T.Compose([T.Resize(224)])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image =  read_image(os.path.join(self.data, self.images[index]))
        image = self.transform(image) / 255
        label = self.labels[index]
        return image, label
        
if __name__ == '__main__':
    # h5_to_jpeg('data\pcam')
    dataset = HE_Dataset('data/pcam/train')
    # dataset = HE_Dataset('data/pcam/test')
    # dataset = HE_Dataset('data/pcam/valid')
    x, y = dataset.__getitem__(1)
    print(x.shape, y)
    print(x.dtype, y)
    # plt.imshow(x.permute(1, 2, 0))
    # plt.show()