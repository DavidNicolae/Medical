import os
import cv2
import json
import torch
import shutil
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.models as models
from torch import nn
from utils import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
# from sklearn.metrics import confusion_matrix

DATA = ['data/pcam/trainHE', 'data/TMA/train']
RESULTS_RIGHT = 'data/kmeans/right'
RESULTS_WRONG = 'data/kmeans/wrong'

class MyDataset(Dataset):
    def __init__(self, images, domain_labels):
        self.images = images
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.domain_labels[index]

def shuffle_data(images, image_names, labels, labels_by_class):
    indices = torch.randperm(len(images))

    images[:] = [images[i] for i in indices.tolist()]
    image_names[:] = [image_names[i] for i in indices.tolist()]
    labels[:] = labels[indices]
    labels_by_class[:] = labels_by_class[indices]

def get_aggregate(images):
    aggregate = np.mean(images, axis=0)
    return aggregate

def reinhard_normalization(source_lab, target_lab):
    # Compute the mean and standard deviation for each channel in both images
    source_mean, source_std = np.mean(source_lab, axis=(0, 1)), np.std(source_lab, axis=(0, 1))
    target_mean, target_std = np.mean(target_lab, axis=(0, 1)), np.std(target_lab, axis=(0, 1))
    
    # Normalize the source image's channels
    normalized_source = (source_lab - source_mean) / source_std
    
    # Apply the target image's mean and standard deviation
    transformed_source = normalized_source * target_std + target_mean

    # Convert the result back to the RGB color space
    transformed_rgb = cv2.cvtColor(transformed_source.astype(np.uint8), cv2.COLOR_Lab2BGR)

    return transformed_rgb

def normalize(images):
    lab_images = [cv2.cvtColor(img, cv2.COLOR_BGR2Lab) for img in images]
    aggregate = get_aggregate(lab_images[:len(lab_images) // len(DATA)])
    images = [reinhard_normalization(img, aggregate) for img in images]
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    images = [torch.from_numpy((img / 255 - mean) / std).float().permute(2, 0, 1) for img in images]
    return images

def get_samples(n_samples):
    all_images = []
    image_names = []
    all_labels = []
    labels_by_class = []
    
    for dir in DATA:
        neg = n_samples
        files = os.listdir(dir)
        labels_file = files[-1]
        f = open(os.path.join(dir, labels_file), 'r')
        labels = list(json.load(f).values())
        for label in labels:
            if neg > 0:
                if label[1] == 0:
                    img = cv2.imread(os.path.join(dir, label[0]))
                    neg -= 1
                    all_images.append(img)
                    all_labels.append(0)
                    image_names.append(os.path.join(dir, label[0]))
            else:
                break
    
    for i in range(len(DATA)):
        labels_by_class += [i] * n_samples
    
    return all_images, image_names, np.array(all_labels), np.array(labels_by_class)

# def get_feature_vectors(features):
#     half = len(features) // len(DATA)
#     mean_feature_vector1 = torch.mean(features[:half], axis=0)
#     mean_feature_vector2 = torch.mean(features[half:], axis=0)
    
#     return [mean_feature_vector1, mean_feature_vector2]

def domain_adaptation_loss(logits, labels):
    return nn.BCELoss()(logits, labels)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(torch.nn.Module):
    def __init__(self, alpha):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class Encoder(pl.LightningModule):
    def __init__(self, alpha=1.0):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        layers = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        self.grl = GradientReversal(alpha)
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        domain_logits = self.domain_classifier(self.grl(features_flat))
        return features_flat, domain_logits.squeeze()

# def get_features(images, n_samples, batch_size):
#     features = []
#     encoder = Encoder().to('cuda')
#     for i in range(0, n_samples, batch_size):
#         batch = torch.stack(images[i:min(i + batch_size, n_samples)])
#         feature_vector = encoder(batch.to(encoder.device))
#         features.append(feature_vector.cpu().view(feature_vector.size(0), -1))
#     features = torch.cat(features)
#     feature_vectors = get_feature_vectors(features)
#     # feature_vectors = PCA(2).fit_transform(feature_vectors)
#     return features, feature_vectors

def train_domain_adaptation(encoder, train_loader, val_loader, epochs):
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    for epoch in range(epochs):
        encoder.train()
        for images, domain_labels in train_loader:
            optimizer.zero_grad()
            _, domain_logits = encoder(images.to(encoder.device))
            loss = domain_adaptation_loss(domain_logits, domain_labels.to(encoder.device))
            loss.backward()
            optimizer.step()

        encoder.eval()
        val_loss = 0
        for images, domain_labels in val_loader:
            _, domain_logits = encoder(images.to(encoder.device))
            val_loss += domain_adaptation_loss(domain_logits, domain_labels.to(encoder.device)).item()

        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}")


class Kmeans:
    def __init__(self, features, image_names, labels, labels_by_class, n_samples, n_clusters, state):
        # initial_centers = PCA(2).fit_transform(np.stack(feature_vectors))
        self.kmeans = KMeans(n_clusters, random_state=state, n_init='auto')
        self.features = features
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
        self.mapped_clusters = [cluster_to_label_map[c] for c in self.clusters]
    
    def accuracy(self):
        return accuracy_score(self.labels_by_class, self.mapped_clusters)
    
    def visualize(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Scatter plot of clusters
        axes[0, 0].scatter(self.features[:, 0], self.features[:, 1], c=self.clusters, cmap='viridis', alpha=0.5)
        axes[0, 0].scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)
        axes[0, 0].set_title('K-means Clustering with PCA')

        # Scatter plot of original data labels
        scatter_kwargs = dict(facecolor='none', alpha=0.5)
        axes[0, 1].scatter([self.features[idx, 0] for idx, lbl in enumerate(labels_by_class) if lbl == 0],
                           [self.features[idx, 1] for idx, lbl in enumerate(labels_by_class) if lbl == 0],
                           edgecolors='r', label='Pcam', s=10, **scatter_kwargs)
        axes[0, 1].scatter([self.features[idx, 0] for idx, lbl in enumerate(labels_by_class) if lbl == 1],
                           [self.features[idx, 1] for idx, lbl in enumerate(labels_by_class) if lbl == 1],
                           edgecolors='b', label='TMA', s=10, **scatter_kwargs)
        axes[0, 1].scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], c='black', marker='x', s=100, label='Centroids')
        axes[0, 1].set_title('K-means Clustering with PCA')
        axes[0, 1].legend()

        # 2D histogram (density plot)
        x, y = self.features[:, 0], self.features[:, 1]
        bins = 50
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        pcm = axes[1, 0].pcolormesh(x_edges, y_edges, hist.T, cmap='viridis')
        fig.colorbar(pcm, ax=axes[1, 0], label='Density')
        axes[1, 0].set_title('Density Plot')

        # Clear the unused subplot
        axes[1, 1].axis('off')

        # Set common labels for both rows and columns
        for ax in axes[:, 0]:
            ax.set_ylabel('PC2')
        for ax in axes[0, :]:
            ax.set_xlabel('PC1')

        # Adjust the layout and show the figure
        plt.tight_layout()
        plt.show()
        
    def eval(self):
        self.predict()
        self.map_clusters()
        print('Inertia: ', self.kmeans.inertia_)
        print('Silhouette score: ', silhouette_score(self.features, self.clusters))
        print("Adjusted Rand Index: ", adjusted_rand_score(self.labels_by_class, self.clusters))
        print('Accuracy: ', self.accuracy())
        
    def save_cluster_mapping(self):
        classified_right = [i for i, (y_h, y) in enumerate(zip(self.mapped_clusters, self.labels_by_class)) if y_h == y]
        # right_labels = [self.labels[i] for i in classified_right]
        classified_wrong = [i for i, (y_h, y) in enumerate(zip(self.mapped_clusters, self.labels_by_class)) if y_h != y]
        # wrong_labels = [self.labels[i] for i in classified_wrong]
        
        if not os.path.isdir(RESULTS_RIGHT):
            os.makedirs(RESULTS_RIGHT)
        if not os.path.isdir(RESULTS_WRONG):
            os.makedirs(RESULTS_WRONG)
        
        for idx in classified_right:
            image = self.image_names[idx]
            shutil.copy(image, os.path.join(RESULTS_RIGHT, 'img_' + str(idx) + '.jpeg'))
            
        for idx in classified_wrong:
            image = self.image_names[idx]
            shutil.copy(image, os.path.join(RESULTS_WRONG, 'img_' + str(idx) + '.jpeg'))

if __name__ == '__main__':
    n_samples = 2500
    n_clusters = 2
    state = 0
    batch_size = 128
    torch.seed()
    
    images, image_names, labels, labels_by_class = get_samples(n_samples)
    images = normalize(images)
    
    # shuffle_data(images, image_names, labels, labels_by_class)
    
    train_ratio = 0.8
    train_size = int(train_ratio * len(images))
    val_size = len(images) - train_size
    data = list(zip(images, labels_by_class.astype(np.float32)))
    train_data, val_data = random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    encoder = Encoder()
    epochs = 50
    train_domain_adaptation(encoder, train_loader, val_loader, epochs)
    
    features = []
    for batch in DataLoader(images, batch_size):
        batch_features, _ = encoder(batch.to(encoder.device))
        reversed_features = encoder.grl(batch_features.to(encoder.device))
        features.append(reversed_features.detach().cpu())
    features = torch.cat(features)
    # features, feature_vectors = get_features(images, n_samples * len(DATA), batch_size)
    
    # rfe(features, labels_by_class)
    # rfe_ranking, rfe_support = get_rfe()
    # features = features[:, np.where(rfe_ranking == 1)[0]]
    
    features = PCA(2).fit_transform(features)
    kmeans = Kmeans(features, image_names, labels, labels_by_class, n_samples * len(DATA), n_clusters, state)
    kmeans.eval()
    
    kmeans.save_cluster_mapping()
    kmeans.visualize()
    