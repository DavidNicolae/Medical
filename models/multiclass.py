import os
import numpy as np
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.utils.data as data
import pytorch_lightning as pl
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.notebook import tqdm
from torchvision import transforms
from data_loader import HE_Dataset
from torchmetrics.functional import accuracy
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torchvision.transforms import ConvertImageDtype
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights

LOGS = 'lightning_logs'
CHECKPOINTS = 'lightning_logs/checkpoints7'

class MixedTransformDataset(data.Dataset):
    """
    A dataset wrapper that applies different transforms depending on the image source.
    Assumes PCam images have 'pcam' in their path, others are treated as non-PCam.
    """
    def __init__(self, base_dataset, pcam_transform, other_transform):
        self.base_dataset = base_dataset
        self.pcam_transform = pcam_transform
        self.other_transform = other_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if 'pcam' in self.base_dataset.labels[str(idx)][0]:
            if self.pcam_transform:
                image = self.pcam_transform(image)
        else:
            if self.other_transform:
                image = self.other_transform(image)
        return image, label

class PCAM_Model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 3)
        self.loss_module = nn.CrossEntropyLoss()
        self.lr = lr
        
        self.save_hyperparameters()
        
        self.precision
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, patience=4, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}
    
    def training_step(self, batch):
        x, y = batch
        y_h = self(x)
        preds = y_h.argmax(dim=1)
        loss = self.loss_module(y_h, y)
        train_accuracy = accuracy(preds, y, num_classes=3, task='multiclass')
        self.log('train_accuracy', train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        preds = y_h.argmax(dim=1)
        loss = self.loss_module(y_h, y)
        valid_accuracy = accuracy(preds, y, num_classes=3, task='multiclass')
        self.log('valid_accuracy', valid_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        preds = y_h.argmax(dim=1)
        loss = self.loss_module(y_h, y)
        test_accuracy = accuracy(preds, y, num_classes=3, task='multiclass')
        self.log('test_accuracy', test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
if __name__ == '__main__':
    pl.seed_everything(42)

    lr = 0.0001
    epochs = 100
    batch_size = 64
    resume = False
    train = True
    k_folds = 8

    # PCam-specific transform: random affine to shift content, then center crop
    pcam_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # up to 20% shift
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5766, 0.4289, 0.7023], std=[0.1904, 0.2059, 0.1479])
    ])

    # Other dataset transform (as before)
    other_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5766, 0.4289, 0.7023], std=[0.1904, 0.2059, 0.1479])
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5766, 0.4289, 0.7023], std=[0.1904, 0.2059, 0.1479])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5766, 0.4289, 0.7023], std=[0.1904, 0.2059, 0.1479])
    ])

    checkpoint_loss_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS,
        filename='pcam-{epoch:02d}-{valid_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='valid_loss',
        mode='min'
    )
    
    checkpoint_acc_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS,
        filename='pcam-{epoch:02d}-{valid_accuracy:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='valid_accuracy',
        mode='max'
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='valid_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min'
    )

    # Use the base dataset with no transform, then wrap with MixedTransformDataset
    base_train_dataset = HE_Dataset('data/mixed/train', 'labels.json', transforms=None)
    train_dataset = MixedTransformDataset(base_train_dataset, pcam_train_transform, other_train_transform)
    train_loader = data.DataLoader(train_dataset, batch_size, pin_memory=True, num_workers=5, shuffle=True)

    valid_dataset = HE_Dataset('data/mixed/valid', 'labels.json', valid_transform)
    valid_loader = data.DataLoader(valid_dataset, batch_size, pin_memory=True, num_workers=2)

    test_dataset = HE_Dataset('data/mixed/test', 'labels.json')
    test_loader = data.DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=2, shuffle=True)
    
    checkpoints = os.listdir(CHECKPOINTS)
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=-1,
                         callbacks=[checkpoint_loss_callback, checkpoint_acc_callback, early_stopping_callback])
    
    if train:
        pcam = PCAM_Model(lr)
        print('Starting...\n')
        print(f'lr = {lr}')

        lr_finder = trainer.tuner.lr_find(pcam, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        print(f'lr suggestion = {new_lr}')
        pcam.lr = float(input('choose learning rate\n lr = '))
        
        trainer.fit(pcam, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    elif resume:
        pcam = PCAM_Model.load_from_checkpoint(checkpoint_path=os.path.join(CHECKPOINTS, checkpoints[1]))
        print('Resuming training...\n')
        trainer.fit(pcam, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print('Testing...\n')
    
    
    for checkpoint in os.listdir(CHECKPOINTS):
        pcam = PCAM_Model.load_from_checkpoint(checkpoint_path=os.path.join(CHECKPOINTS, checkpoint))
        gradcam = GradCAM(model=pcam, target_layers=[pcam.backbone.layer4[-1]], use_cuda=True)
        pcam.to(torch.device('cuda'))
        pcam.eval()
        
        all_labels = []
        all_preds = []
        vis_maps = []
        vis_labels = []
        lbl_1_0 = 0
        lbl_2_0 = 0
        lbl_0_2 = 0
        lbl_0_1 = 0
        cond = True
        
        for x, y in test_loader:
            x_tr = torch.stack([test_transform(xi) for xi in x]).to(pcam.device)
            preds = pcam(x_tr).argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(y)
            
            targets = [ClassifierOutputTarget(i) for i in preds]
            grayscale_cam = gradcam(input_tensor=x_tr, targets=targets, aug_smooth=True, eigen_smooth=True)
            rgb_images = x.numpy()
            
            if cond:
                fig, axs = plt.subplots(4, 4, figsize=(8, 8))
                fig.suptitle('Prediction | Ground Truth', fontsize=16, fontweight='bold')
                for i, ax in enumerate(axs.flatten()):
                    if i >= len(rgb_images):
                        break
                    ax.imshow(rgb_images[i].transpose((1, 2, 0)))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if preds[i] != y[i]:
                        title_color = 'red'
                    else:
                        title_color = 'black'
                    title = f'{preds[i]} | {y[i]}'
                    ax.set_title(title, color=title_color, fontsize=16, pad=2)
                    
                fig.subplots_adjust(hspace=0.2, wspace=0.2)


                plt.show()
                cond = False
            
            for i in range(len(rgb_images)):
                single_rgb_img = rgb_images[i].transpose((1, 2, 0)) / 255
                single_grayscale_cam = grayscale_cam[i]
                visualization = show_cam_on_image(single_rgb_img, single_grayscale_cam, use_rgb=True)
                if len(vis_maps) < 16:
                    if preds[i] == 1 and y[i] == 0 and lbl_1_0 < 4:
                        vis_maps.append(visualization)
                        lbl_1_0 += 1
                    elif preds[i] == 2 and y[i] == 0 and lbl_2_0 < 4:
                        lbl_2_0 += 1
                        vis_maps.append(visualization)
                    elif preds[i] == 0 and y[i] == 1 and lbl_0_1 < 4:
                        lbl_0_1 += 1
                        vis_maps.append(visualization)
                    elif preds[i] == 0 and y[i] == 2 and lbl_0_2 < 4:
                        lbl_0_2 += 1
                        vis_maps.append(visualization)
            
        all_preds = torch.cat(all_preds).cpu()
        all_labels = torch.cat(all_labels)
        
        report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive breast", "Positive lymphoma"], digits=3)
        print(report)
        
        conf_matrix = confusion_matrix(all_labels, all_preds)

        fig, ax = plt.subplots()
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Set up axes
        labels = ["Negative", "Breast", "Lymphoma"]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=16)
        ax.set_yticklabels(labels, fontsize=16)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # Loop over data dimensions and create text annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black", fontsize=20)

        plt.xlabel('Prediction', fontsize=16)
        plt.ylabel('Ground Truth', fontsize=16)
        plt.show()
        
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axs.flatten()):
            if i >= len(vis_maps):
                break
            ax.imshow(vis_maps[i])
            ax.set_xticks([])
            ax.set_yticks([])

        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        plt.show()
        break