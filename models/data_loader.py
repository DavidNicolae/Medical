import os
import torch.utils.data as data
import torchvision.transforms as T
import json
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision import transforms

class HE_Dataset(data.Dataset):
    
    def __init__(self, data_dir, labels_file, transforms=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        f = open(os.path.join(data_dir, labels_file), 'r')
        self.labels = json.load(f)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image =  read_image(os.path.join(self.data_dir, self.labels[str(index)][0]))
        if self.transforms:
            image = self.transforms(image)
        label = self.labels[str(index)][1]
        return image, label
        
if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = HE_Dataset('data/pcam/trainHE', 'labels.json', transform)
    x, y = dataset.__getitem__(0)
    print(dataset.__len__())
    print(x, y)
    plt.imshow(x.permute(1, 2, 0))
    plt.show()
    
    f = open(os.path.join('data/pcam/testHE', 'labels.json'), 'r')
    labels = json.load(f)
    labels = [val[1] for val in list(labels.values())]
    print(labels.count(1), labels.count(0), len(labels))