import os
import h5py
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

DIR = 'data'
FILE_FORMAT = 'camelyonpatch_level_2_split_{}_{}.h5'

class HE_Dataset(data.Dataset):
    
    def __init__(self, stage='train') -> None:
        super().__init__()
        self.load(stage)

    def load(self, stage):  
        self.images = h5py.File(os.path.join(DIR, FILE_FORMAT.format(stage, 'x')), 'r')
        self.labels = h5py.File(os.path.join(DIR, FILE_FORMAT.format(stage, 'y')), 'r')
        
        self.x_key = list(self.images.keys())[0]
        self.y_key = list(self.labels.keys())[0]
    
    def __len__(self):
        return len(self.images[self.x_key])
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.images[self.x_key][index] / 255).float()
        y = torch.from_numpy(self.labels[self.y_key][index]).flatten().float()
        return x, y
        
if __name__ == '__main__':
    dataset = HE_Dataset()
    x, y = dataset.__getitem__(3)
    print(x.dtype, y.dtype)
    