import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

DATA = ['data/pcam/train', 'data/TMA/train']
# DATA = ['data/TMA/train']

def preprocess(images):
    images = np.array(images).reshape((len(images), -1)) / 255
    return images

def get_samples():
    all_images = []
    all_labels = []
    
    for dir in DATA:
        pos, neg = 500, 500
        files = os.listdir(dir)
        labels_file = files[-1]
        f = open(os.path.join(dir, labels_file), 'r')
        labels = list(json.load(f).values())
        for label in labels:
            if label[1] == 1 and pos > 0:
                img = cv2.imread(os.path.join(dir, label[0]))
                pos -= 1
                all_images.append(img)
                all_labels.append(1)
            elif label[1] == 0 and neg > 0:
                img = cv2.imread(os.path.join(dir, label[0]))
                neg -= 1
                all_images.append(img)
                all_labels.append(0)
            elif pos == 0 and neg == 0:
                break
            
    all_images = preprocess(all_images)
    return all_images, all_labels

class Kmeans:
    def __init__(self, images, labels, n_clusters, state):
        self.kmeans = KMeans(n_clusters, random_state=state, n_init='auto')
        self.images_2d = PCA(2).fit_transform(images)
        self.labels = labels
    
    def predict(self):
        self.clusters = self.kmeans.fit_predict(self.images_2d)
    
    def visualize(self):
        plt.scatter(self.images_2d[:, 0], self.images_2d[:, 1], c=self.clusters, cmap='viridis', alpha=0.5)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)
        plt.title('K-means Clustering with PCA')
        plt.show()
        
        plt.scatter(self.images_2d[:len(self.labels) // 2, 0], self.images_2d[:len(self.labels) // 2, 1],
                    facecolor='none', edgecolors='r', label='Pcam', alpha=0.5)
        plt.scatter(self.images_2d[len(self.labels) // 2:, 0], self.images_2d[len(self.labels) // 2:, 1],
                    facecolor='none', edgecolors='b', label='TMA', alpha=0.5)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1],
                    c='black', marker='x', s=100, label='Centroids')
        plt.title('K-means Clustering with PCA')
        plt.legend()
        plt.show()
        
        # fig, ax = plt.subplots()
        # hb = ax.hexbin(self.images_2d[:, 0], self.images_2d[:, 1], cmap='viridis', gridsize=50)
        # cb = fig.colorbar(hb, ax=ax)
        # cb.set_label('Density')
        # ax.set_xlabel('Component 1')
        # ax.set_ylabel('Component 2')
        # plt.title('Hexbin Plot with PCA')
        # plt.show()
    
    # def visualize3(self):
    #     print(f'images_shape[1] = {self.images.shape[1]}')
    #     x_data = [i for i in range(self.images.shape[1])]
    #     plt.scatter(x_data, self.kmeans.cluster_centers_[0], c='red', alpha=0.2, s=70)
    #     plt.scatter(x_data, self.kmeans.cluster_centers_[1], c='blue', alpha=0.2, s=50)
    #     plt.show()
    

if __name__ == '__main__':
    images, labels = get_samples()
    kmeans = Kmeans(images, labels, 2, 0)
    kmeans.predict()
    kmeans.visualize()