import os
import h5py
from torch import Tensor

DIR = 'data'
FILE_FORMAT = 'camelyonpatch_level_2_split_{}_{}.h5'

def load_slice(slice, stage='train'):
    
    images = h5py.File(os.path.join(DIR, FILE_FORMAT.format(stage, 'x')), 'r')
    labels = h5py.File(os.path.join(DIR, FILE_FORMAT.format(stage, 'y')), 'r')
    
    data_x = images[list(images.keys())[0]]
    data_y = labels[list(labels.keys())[0]]
    
    x = Tensor(data_x[slice[0]:slice[1]] / 255)
    y = Tensor(data_y[slice[0]:slice[1]]).flatten()
    
    return x, y

if __name__ == '__main__':
    x, y = load_slice((0, 6))
    print(x[0], y[0].dtype)
