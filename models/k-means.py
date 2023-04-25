import os
import json
import torch
import shutil
import numpy as np
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.models as models
from torch import nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import confusion_matrix

DATA = ['data/pcam/trainHE', 'data/TMA/train']
RESULTS = 'data/kmeans'

def normalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = ((image / 255 - mean) / std).squeeze()
    return image

def get_samples(n_samples):
    all_images = []
    image_names = []
    all_labels = []
    labels_by_class = []
    
    for dir in DATA:
        pos, neg = n_samples // 2, n_samples // 2
        files = os.listdir(dir)
        labels_file = files[-1]
        f = open(os.path.join(dir, labels_file), 'r')
        labels = list(json.load(f).values())
        for label in labels:
            if label[1] == 1 and pos > 0:
                img = torchvision.io.read_image(os.path.join(dir, label[0]))
                pos -= 1
                all_images.append(normalize(img))
                all_labels.append(1)
                image_names.append(os.path.join(dir, label[0]))
            elif label[1] == 0 and neg > 0:
                img = torchvision.io.read_image(os.path.join(dir, label[0]))
                neg -= 1
                all_images.append(normalize(img))
                all_labels.append(0)
                image_names.append(os.path.join(dir, label[0]))
            elif pos == 0 and neg  == 0:
                break
    
    for i in range(len(DATA)):
        labels_by_class += [i] * n_samples
    return all_images, image_names, np.array(all_labels), np.array(labels_by_class)

class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        layers = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return features
            
class Kmeans:
    def __init__(self, images, image_names, labels, labels_by_class, n_samples, n_clusters, state, features=False, batch_size = 256):
        self.kmeans = KMeans(n_clusters, random_state=state, n_init='auto')
        if features:
            features = []
            encoder = Encoder().to('cuda')
            for i in range(0, n_samples, batch_size):
                batch = torch.stack(images[i:min(i + batch_size, n_samples)])
                feature_vector = encoder(batch.to(encoder.device))
                features.append(feature_vector.cpu().view(feature_vector.size(0), -1))
            features = torch.cat(features)
        else:
            features = torch.stack(images).reshape((n_samples, -1))
        self.features = PCA(2).fit_transform(features)
        self.image_names = image_names
        self.labels = labels
        self.labels_by_class = labels_by_class
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        
    def predict(self):
        self.clusters = self.kmeans.fit_predict(self.features)
    
    def map_clusters(self):
        n_classes = len(np.unique(self.labels_by_class))
        cost_matrix = np.zeros((self.n_clusters, n_classes))
        
        for i in range(self.n_clusters):
            for j in range(n_classes):
                cost_matrix[i][j] = np.sum((self.clusters == i) & (self.labels_by_class == j))
        
        print(cost_matrix)
        row, col = linear_sum_assignment(cost_matrix, maximize=True)
        cluster_to_label_map = dict(zip(row, col))
        self. mapped_clusters = [cluster_to_label_map[c] for c in self.clusters]
    
    def accuracy(self):
        return accuracy_score(self.labels_by_class, self.mapped_clusters)
    
    def visualize(self):
        plt.scatter(self.features[:, 0], self.features[:, 1], c=self.clusters, cmap='viridis', alpha=0.5)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)
        plt.title('K-means Clustering with PCA')
        plt.show()
        
        plt.scatter(self.features[:len(self.labels) // 2, 0], self.features[:len(self.labels) // 2, 1],
                    facecolor='none', edgecolors='r', label='Pcam', alpha=0.5)
        plt.scatter(self.features[len(self.labels) // 2:, 0], self.features[len(self.labels) // 2:, 1],
                    facecolor='none', edgecolors='b', label='TMA', alpha=0.5)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1],
                    c='black', marker='x', s=100, label='Centroids')
        plt.title('K-means Clustering with PCA')
        plt.legend()
        plt.show()
    
    def eval(self):
        self.predict()
        self.map_clusters()
        print('Inertia: ', self.kmeans.inertia_)
        print('Silhouette score: ', silhouette_score(self.features, self.clusters))
        print("Adjusted Rand Index: ", adjusted_rand_score(self.labels_by_class, self.clusters))
        print('Accuracy: ', self.accuracy())
        self.visualize()
        
    def get_cluster_mapping(self):
        classified_right = [i for i, (y_h, y) in enumerate(zip(self.mapped_clusters, self.labels_by_class)) if y_h == y]
        right_labels = [self.labels[i] for i in classified_right]
        # classified_wrong = [i for i, (y_h, y) in enumerate(zip(self.mapped_clusters, self.labels_by_class)) if y_h != y]
        # wrong_labels = [self.labels[i] for i in wrong]
        
        if not os.path.isdir(RESULTS):
            os.makedirs(RESULTS)
        
        for idx in classified_right:
            image = self.image_names[idx]
            shutil.copy(image, os.path.join(RESULTS, 'img_' + str(idx) + '.jpeg'))

if __name__ == '__main__':
    n_samples = 1000
    n_clusters = 2
    state = 0
    features = True
    
    images, image_names, labels, labels_by_class = get_samples(n_samples)
    kmeans = Kmeans(images, image_names, labels, labels_by_class, n_samples * len(DATA), n_clusters, 0, features)
    kmeans.eval()
    
    right = kmeans.get_cluster_mapping()
    