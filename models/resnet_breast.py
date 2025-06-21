import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.notebook import tqdm
from torchvision import transforms
from data_loader import HE_Dataset
from torchmetrics.functional import accuracy
from torchvision.transforms import ConvertImageDtype
from torchmetrics import AUROC
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

LOGS = 'lightning_logs'
CHECKPOINTS = 'lightning_logs/checkpoints'

class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv2=nn.Sequential(
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv3=nn.Sequential(
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv4=nn.Sequential(
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv5=nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        
        self.dropout2d = nn.Dropout2d()
        
        
        self.fc=nn.Sequential(
                nn.Linear(512*3*3,1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(1024,512),
                nn.Dropout(0.4),
                nn.Linear(512, 1),
                nn.Sigmoid())
        
        self.loss_module = nn.BCELoss()
        self.train_auroc = AUROC(task='binary', num_classes=2)
        self.valid_auroc = AUROC(task='binary', num_classes=2)
        self.test_auroc = AUROC(task='binary', num_classes=2)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00015)
        return optimizer
    
    def training_step(self, batch):
        x, y = batch
        y = y.float()
        y_h = self(x)
        print(y_h)
        loss = self.loss_module(y_h, y)
        train_accuracy = accuracy(y_h.round(), y, task='binary')
        self.log('train_accuracy', train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train_precision', self.train_precision(y_h, y), prog_bar=True)
        # self.log('train_recall', self.train_recall(y_h, y), prog_bar=True)
        # self.log('train_f1', self.train_f1(y_h, y), prog_bar=True)
        # self.log('train_auroc', self.train_auroc(y_h, y), prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y = y.float()
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        valid_accuracy = accuracy(y_h.round(), y, task='binary')
        self.log('valid_accuracy', valid_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        # self.log('valid_precision', self.valid_precision(y_h, y), prog_bar=True)
        # self.log('valid_recall', self.valid_recall(y_h, y), prog_bar=True)
        # self.log('valid_f1', self.valid_f1(y_h, y), prog_bar=True)
        self.log('valid_auroc', self.valid_auroc(y_h, y), on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        y = y.float()
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        test_accuracy = accuracy(y_h.round(), y, task='binary')
        self.log('test_accuracy', test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_precision', self.test_precision(y_h, y), prog_bar=True)
        # self.log('test_recall', self.test_recall(y_h, y), prog_bar=True)
        # self.log('test_f1', self.test_f1(y_h, y), prog_bar=True)
        self.log('test_auroc', self.test_auroc(y_h, y), on_epoch=True, prog_bar=True)
        return loss
    
class PCAM_Model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            # nn.Dropout(p=0.1),
            nn.Sigmoid()
        )
        self.loss_module = nn.BCELoss()
        self.lr = lr
        
        # self.backbone.eval()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
            
        # for param in self.backbone.fc.parameters():
        #     param.requires_grad = True
        
        self.save_hyperparameters()
        
        self.train_auroc = AUROC(task='binary', num_classes=2)
        self.valid_auroc = AUROC(task='binary', num_classes=2)
        self.test_auroc = AUROC(task='binary', num_classes=2)
        
    def forward(self, x):
        x = self.backbone(x)
        return x.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}
    
    def training_step(self, batch):
        x, y = batch
        y = y.float()
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        train_accuracy = accuracy(y_h.round(), y, task='binary')
        self.log('train_accuracy', train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        y = y.float()
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        valid_accuracy = accuracy(y_h.round(), y, task='binary')
        self.log('valid_accuracy', valid_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('valid_auroc', self.valid_auroc(y_h, y), on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, _):
        x, y = batch
        y = y.float()
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        test_accuracy = accuracy(y_h.round(), y, task='binary')
        self.log('test_accuracy', test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_auroc', self.test_auroc(y_h, y), on_epoch=True, prog_bar=True)
    
if __name__ == '__main__':

    lr = 0.0001
    epochs = 100
    batch_size = 64
    resume = False
    train = False
    k_folds = 8

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.601, 0.383, 0.684], std=[0.186, 0.184, 0.147])
    ])
    
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.601, 0.383, 0.684], std=[0.186, 0.184, 0.147])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.601, 0.383, 0.684], std=[0.186, 0.184, 0.147])
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
        patience=10,
        verbose=False,
        mode='min'
    )
    
    train_dataset = HE_Dataset('data/pcam/trainHE', 'labels.json', train_transform)
    train_loader = data.DataLoader(train_dataset, batch_size, pin_memory=True, num_workers=10, shuffle=True)

    valid_dataset = HE_Dataset('data/pcam/validHE', 'labels.json', valid_transform)
    valid_loader = data.DataLoader(valid_dataset, batch_size, pin_memory=True, num_workers=5)

    test_dataset = HE_Dataset('data/pcam/testHE', 'labels.json', test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size, pin_memory=True, num_workers=5)
    
    checkpoints = os.listdir(CHECKPOINTS)
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=-1,
                         callbacks=[checkpoint_loss_callback, checkpoint_acc_callback, early_stopping_callback])
    
    if train:
        pcam = PCAM_Model(lr)
        # print('Starting...\n')
        # print(f'lr = {lr}')

        # lr_finder = trainer.tuner.lr_find(pcam, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        # new_lr = lr_finder.suggestion()
        # print(f'lr suggestion = {new_lr}')
        # pcam.lr = float(input('choose learning rate\n lr = '))
        
        trainer.fit(pcam, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    else:
        pcam = PCAM_Model.load_from_checkpoint(checkpoint_path=os.path.join(CHECKPOINTS, checkpoints[1]))
        if resume:
            print('Resuming training...\n')
            trainer.fit(pcam, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print('Testing...\n')
    
    for checkpoint in os.listdir(CHECKPOINTS):
        pcam = PCAM_Model.load_from_checkpoint(checkpoint_path=os.path.join(CHECKPOINTS, checkpoint))
        pcam.to(torch.device('cuda'))
        pcam.eval()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for x, y in test_loader:
                preds = pcam(x.to(pcam.device)).round()
                all_preds.append(preds)
                all_labels.append(y)
        
        all_preds = torch.cat(all_preds).cpu()
        all_labels = torch.cat(all_labels).cpu()
        precision = precision_score(all_labels, all_preds, average=None)
        recall = recall_score(all_labels, all_preds, average=None)
        f1 = f1_score(all_labels, all_preds, average=None)
        roc = roc_auc_score(all_labels, all_preds, average=None)
        
        print(f'Checkpoint: {checkpoint}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'AUC Score: {roc}\n')
        
        report = classification_report(all_labels, all_preds, target_names=["Negative Breast", "Positive Breast"], digits=3)
        print(report)