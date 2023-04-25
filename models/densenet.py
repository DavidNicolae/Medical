import os
import torch
import torchvision
import numpy as np
import torch.utils.data as data
import pytorch_lightning as pl
from torch import nn
from tqdm.notebook import tqdm
from torchvision.models import densenet121, DenseNet121_Weights
from torchmetrics.functional import accuracy
from data_loader import HE_Dataset, h5_to_jpeg

LOGS = 'lightning_logs'

class PCAM_Model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.loss_module = nn.CrossEntropyLoss()
        self.lr = lr
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 2)
        self.save_hyperparameters()
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
    
    def training_step(self, batch):
        x, y = batch
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        y_h = y_h.argmax(dim=1)
        train_accuracy = accuracy(y_h, y, task='binary')
        self.log('train_accuracy', train_accuracy, prog_bar=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'train_accuracy': train_accuracy}
    
    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        y_h = y_h.argmax(dim=1)
        test_accuracy = accuracy(y_h, y, task='binary')
        self.log('test_accuracy', test_accuracy, prog_bar=True, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'test_accuracy': test_accuracy}
    
if __name__ == '__main__':
    
    pcam = PCAM_Model(0.01)
    # h5_to_jpeg('data\pcam')
    # pl.seed_everything(42)
    
    # lr = 0.001
    # loss = nn.CrossEntropyLoss()
    
    # if os.path.exists(LOGS):
    #     versions = os.listdir(LOGS)
    #     latest = os.path.join(LOGS, versions[-1], 'checkpoints')
    #     checkpoint = os.listdir(latest)[0]
    #     checkpoint = os.path.join(latest, checkpoint)
    #     pcam = PCAM_Model.load_from_checkpoint(checkpoint)
    #     print('Loaded model from latest checkpoint: ', checkpoint)
    # else:
    #     pcam = PCAM_Model(lr)
    #     print('Initialized new model')
    
    train_dataset = HE_Dataset('data/pcam/train', 'labels.json')
    train_loader = data.DataLoader(train_dataset, batch_size=128, pin_memory=True, num_workers=5, shuffle=True)

    # test_dataset = HE_Dataset('data/pcam/test', 'labels.json')
    # test_loader = data.DataLoader(test_dataset, batch_size=128, pin_memory=True, num_workers=5)
    
    trainer = pl.Trainer(max_epochs=5, accelerator='gpu', devices=-1)
    
    trainer.fit(pcam, train_dataloaders=train_loader)
    
    # trainer.test(pcam, test_loader)