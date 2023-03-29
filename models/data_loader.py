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
        labels_dict = {}
        
        for index, image in enumerate(images[x_key]):
            img = Image.fromarray(image)
            img_name = 'img_' + str(index) + '.jpeg'
            img.save(os.path.join(inner_path, img_name), 'jpeg')
            labels_dict[index] = (img_name, labels[y_key][index].flatten().tolist()[0])
            
        with open(os.path.join(inner_path, 'labels.json'), 'w') as f:
            json.dump(labels_dict, f, indent=4)

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
    h5_to_jpeg('data\pcam')
    dataset = HE_Dataset('data/pcam/train', 'labels.json')
    # dataset = HE_Dataset('data/pcam/test')
    # dataset = HE_Dataset('data/pcam/valid')
    x, y = dataset.__getitem__(0)
    print(dataset.__len__())
    print(x, y)
    print(x.shape, y)
    print(x.dtype, y)
    plt.imshow(x.permute(1, 2, 0))
    plt.show()