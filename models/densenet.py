from tqdm.notebook import tqdm
import torch
from torch import nn
from torchmetrics.functional import accuracy
import torch.utils.data as data
import pytorch_lightning as pl
from data_loader import HE_Dataset

class PCAM_Model(pl.LightningModule):
    def __init__(self, model, loss_module, lr) -> None:
        super().__init__()
        self.model = model
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(1024, 2)
        
        self.loss_module = loss_module
        self.lr = lr
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
    
    def training_step(self, batch):
        x, y = batch
        y_h = self(x)
        loss = self.loss_module(y_h, y.flatten())
        y_h = y_h.argmax(dim=1, keepdim=True).float()
        train_accuracy = accuracy(y_h, y, task='binary')
        self.log('train_accuracy', train_accuracy, prog_bar=True)
        self.log('train_loss', loss)
        return {'loss': loss, 'train_accuracy': train_accuracy}
    
    def test_step(self, batch):
        x, y = batch
        y_h = self(x)
        loss = self.loss_module(y_h, y)
        test_accuracy = accuracy(y_h.argmax(dim=1), y)
        return {'test_loss': loss, 'test_accuracy': test_accuracy}
    
    def on_test_epoch_end(self, outputs):
        test_outs = [test_out['test_accuracy'] for test_out in outputs]
        total_accuracy = torch.stack(test_outs).mean()
        self.log('total_test_accuracy', total_accuracy, on_epoch=True, on_step=False)
        return total_accuracy
            
        
if __name__ == '__main__':
    densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121' , weights='DenseNet121_Weights.DEFAULT')
    
    lr = 0.01
    loss = nn.CrossEntropyLoss()
    pcam = PCAM_Model(densenet, loss, lr)
    
    train_dataset = HE_Dataset()
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=20)

    test_dataset = HE_Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=32, num_workers=20)
    
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=-1)
    trainer.fit(pcam, train_dataloaders=train_loader)