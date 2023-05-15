import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch import nn
from tqdm.notebook import tqdm
from torchvision import transforms
from data_loader import HE_Dataset
from torchmetrics.functional import accuracy
from torchvision.transforms import ConvertImageDtype
from torchmetrics import Precision, Recall, F1Score, AUROC
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import resnet50, ResNet50_Weights

LOGS = 'lightning_logs'

class PCAM_Model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.loss_module = nn.CrossEntropyLoss()
        self.lr = lr
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        self.save_hyperparameters()
        
        self.train_precision = Precision(num_classes=2, average='macro')
        self.train_recall = Recall(num_classes=2, average='macro')
        self.train_f1 = F1Score(num_classes=2, average='macro')
        self.train_auroc = AUROC(num_classes=2)

        self.test_precision = Precision(num_classes=2, average='macro')
        self.test_recall = Recall(num_classes=2, average='macro')
        self.test_f1 = F1Score(num_classes=2, average='macro')
        self.test_auroc = AUROC(num_classes=2)
        
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
        self.log('train_precision', self.train_precision(y_h, y), prog_bar=True)
        self.log('train_recall', self.train_recall(y_h, y), prog_bar=True)
        self.log('train_f1', self.train_f1(y_h, y), prog_bar=True)
        self.log('train_auroc', self.train_auroc(y_h, y), prog_bar=True)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        y_h = y_h.argmax(dim=1)
        test_accuracy = accuracy(y_h, y, task='binary')
        self.log('test_accuracy', test_accuracy, prog_bar=True, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_precision', self.test_precision(y_h, y), prog_bar=True)
        self.log('test_recall', self.test_recall(y_h, y), prog_bar=True)
        self.log('test_f1', self.test_f1(y_h, y), prog_bar=True)
        self.log('test_auroc', self.test_auroc(y_h, y), prog_bar=True)
        return loss
    
if __name__ == '__main__':
    pl.seed_everything(42)

    lr = 0.001
    epochs = 5
    batch_size = 64
    pcam = PCAM_Model(lr)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        ConvertImageDtype(torch.float32)
    ])
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=LOGS,
        filename='pcam-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    train_dataset = HE_Dataset('data/pcam/trainHE', 'labels.json', train_transform)
    train_loader = data.DataLoader(train_dataset, batch_size, pin_memory=True, num_workers=5, shuffle=True)

    test_dataset = HE_Dataset('data/pcam/testHE', 'labels.json', test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=128, pin_memory=True, num_workers=5)
    
    trainer = pl.Trainer(max_epochs=epochs, gpus=-1) # add callbacks param
    
    trainer.fit(pcam, train_dataloaders=train_loader)
    
    trainer.test(pcam, test_loader)