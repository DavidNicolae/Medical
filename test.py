import matplotlib.pyplot as plt
import torchstain
from torchvision import transforms
import numpy as np
from random import shuffle
import cv2
import numpy

d1 = 'data/pcam/train/'
d2 = 'data/TMA/train/'
# # d1 = 'data/TMA/discarded/'

# def normalize(images):
#     target = images[0]
#     T = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x*255)
#     ])
#     normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
#     normalizer.fit(T(target))
#     for i in range(len(images)):
#         images[i], _, _= normalizer.normalize(I=T(images[i]), stains=False)
#         images[i] = images[i].numpy().astype(np.uint8)
    
#     return images

# # import os
# # print(os.listdir(d1))

# # List of image strings
img_list = [d1+'img_0.jpeg',d1+'img_1.jpeg',d1+'img_2.jpeg',d1+'img_14.jpeg',d1+'img_19.jpeg',d1+'img_29.jpeg',d1+'img_43.jpeg',
            d1+'img_55.jpeg',d2+'img_12.jpeg',d2+'img_45.jpeg',d2+'img_5334.jpeg',d2+'img_247.jpeg',d2+'img_618.jpeg',d2+'img_6266.jpeg',d2+'img_5074.jpeg',d2+'img_3983.jpeg']

# # img_list = [d1+'img_0.jpeg',d1+'img_1.jpeg',d1+'img_2.jpeg',d1+'img_3.jpeg',d1+'img_5.jpeg',d1+'img_7.jpeg',d1+'img_8.jpeg',
# #             d1+'img_10.jpeg',d1+'img_17.jpeg',d1+'img_18.jpeg',d1+'img_19.jpeg',d1+'img_24.jpeg',d1+'img_28.jpeg',d1+'img_29.jpeg',d1+'img_38.jpeg',d1+'img_39.jpeg']

# # img_list = [d1+'img_0.jpeg',d1+'img_1.jpeg',d1+'img_2.jpeg',d1+'img_3.jpeg',d1+'img_5.jpeg',d1+'img_7.jpeg',d1+'img_8.jpeg',
# #             d1+'img_10.jpeg',d1+'img_17.jpeg',d1+'img_18.jpeg',d1+'img_19.jpeg',d1+'img_24.jpeg',d1+'img_28.jpeg',d1+'img_29.jpeg',d1+'img_38.jpeg',d1+'img_39.jpeg']

# # img_list = [d1+'img_3.jpeg',d1+'img_41.jpeg',d1+'img_42.jpeg',d1+'img_596.jpeg',d1+'img_1085.jpeg',d1+'img_1118.jpeg',d1+'img_1564.jpeg', d1+'img_1744.jpeg',
# #             d1+'img_2206.jpeg', d1+'img_2212.jpeg', d1+'img_12009.jpeg', d1+'img_1744.jpeg', d1+'img_72080.jpeg', d1+'img_72081.jpeg', d1+'img_22982.jpeg', d1+'img_73341.jpeg']

# # shuffle(img_list)

# # Create a 4x4 grid of subplots
fig, axs = plt.subplots(4, 4, figsize=(8, 8))

# # Iterate through the subplots and plot each image
for i, ax in enumerate(axs.flatten()):
    # Ignore if there are more images than subplots
    if i >= len(img_list):
        break
    img_str = img_list[i]
    # Create a random array of the same shape as the image to use as placeholder data
    img_data = cv2.imread(img_str)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    ax.imshow(img_data)  # Replace this line with your code to read and decode the image string
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust the spacing between subplots
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# Display the grid of images
plt.show()

# target = cv2.imread(img_list[0])
# target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# fig, axs = plt.subplots(4, 4, figsize=(8, 8))
# for i, ax in enumerate(axs.flatten()):
#     # Ignore if there are more images than subplots
#     if i >= len(img_list):
#         break
#     img_str = img_list[i]
#     # Create a random array of the same shape as the image to use as placeholder data
#     img_data = cv2.imread(img_str)
#     img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#     ims = normalize([target, img_data])
#     img_data = ims[1]
#     ax.imshow(img_data)  # Replace this line with your code to read and decode the image string
#     ax.set_xticks([])
#     ax.set_yticks([])

# # Adjust the spacing between subplots
# fig.subplots_adjust(hspace=0.2, wspace=0.2)

# # Display the grid of images
# plt.show()

# Assuming that 'dataset' is your dataset in PyTorch
from models.data_loader import HE_Dataset
import torch.utils.data as data
import torch
import os

def find_identical_images(images, image_names):
    unique_images = {}
    duplicate_names = []

    for i, image in enumerate(images):
        image_hash = hash(image.data.tobytes())
        if image_hash in unique_images:
            duplicate_names.extend([image_names[unique_images[image_hash]], image_names[i]])
            print(duplicate_names)
        else:
            unique_images[image_hash] = i

    return duplicate_names

if __name__ == '__main__':
    # for stri in ['data/TMA/train', 'data/TMA/test', 'data/TMA/valid']:
    #     train_dataset = HE_Dataset(stri, 'labels.json')
    #     train_loader = data.DataLoader(train_dataset, 64, pin_memory=True, num_workers=5, shuffle=True)

    #     mean = 0.
    #     std = 0.
    #     for images, _ in train_loader:
    #         images = images.to(torch.float32)
    #         batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
    #         images = images.view(batch_samples, images.size(1), -1)
    #         mean += images.mean(2).sum(0)
    #         std += images.std(2).sum(0)

    #     mean /= len(train_loader.dataset)
    #     std /= len(train_loader.dataset)

    #     print(mean / 255, std / 255)
    #     break
    
    
    train = [cv2.imread(os.path.join('data/mixed/train', img_name)) for img_name in os.listdir('data/mixed/train')[:-1]]
    train.extend([cv2.imread(os.path.join('data/mixed/test', img_name)) for img_name in os.listdir('data/mixed/test')[:-1]])
    train.extend([cv2.imread(os.path.join('data/mixed/valid', img_name)) for img_name in os.listdir('data/mixed/valid')[:-1]])
    
    img_names = ['img_' + str(idx) for idx in range(len(train))]
    print(len(train), len(img_names))
    # train.append(cv2.imread('data/TMA/train/img_0.jpeg'))
    # img_names.append('img_10000.jpeg')
    
    duplicates = find_identical_images(train, img_names)
    print(duplicates)
    
    # img = cv2.imread('data/TMA/train/img_0.jpeg')
    # plt.imshow(img)
    # plt.show()
    # img = img / 255
    # plt.imshow(img)
    # print(img)
    # plt.show()