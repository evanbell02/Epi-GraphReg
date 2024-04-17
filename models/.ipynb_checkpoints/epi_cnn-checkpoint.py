import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import lightning.pytorch as pl
import math

class EpiCNN(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=1e-3, hidden_channels=64, num_layers=5, ker_size=3):
        super().__init__()
        self.learning_rate = learning_rate
        
        self.cnn = nn.Sequential(nn.Conv1d(in_channels, hidden_channels, ker_size, padding=(ker_size-1)//2))
        for _ in range(num_layers):
            self.cnn.append(nn.InstanceNorm1d(hidden_channels))
            self.cnn.append(nn.Conv1d(hidden_channels, hidden_channels, ker_size, padding=(ker_size-1)//2))
            self.cnn.append(nn.ReLU())
        self.cnn.append(nn.Conv1d(hidden_channels, out_channels, ker_size, padding=(ker_size-1)//2))
        self.cnn.append(nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        return x

    def training_step(self, batch, batch_idx):

        epi, cage = batch
        
        pred_cage = self(epi)
        
        loss = F.mse_loss(pred_cage, cage)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        tensorboard = self.logger.experiment
            
        return loss   

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            epi, cage = batch
            
            pred_cage = self(epi)
            
            loss = F.mse_loss(pred_cage, cage)
            
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
            tensorboard = self.logger.experiment
                
            return loss    

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            epi, cage = batch
            
            pred_cage = self(epi)
            
            loss = F.mse_loss(pred_cage, cage)
            
            self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
            tensorboard = self.logger.experiment
                
            return loss    
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer