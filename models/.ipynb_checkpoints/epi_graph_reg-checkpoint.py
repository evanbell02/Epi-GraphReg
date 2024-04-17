import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import lightning.pytorch as pl
import math
import torch_geometric as pyg
import time

class EpiGraphReg(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=1e-3, hidden_channels=64, num_layers=5, ker_size=3):
        super().__init__()
        self.learning_rate = learning_rate
        
        self.encoder = nn.Sequential(nn.Conv1d(in_channels, hidden_channels, ker_size, padding=(ker_size-1)//2))
        for _ in range(num_layers):
            self.encoder.append(nn.InstanceNorm1d(hidden_channels))
            self.encoder.append(nn.Conv1d(hidden_channels, hidden_channels, ker_size, padding=(ker_size-1)//2))
            self.encoder.append(nn.ReLU())
        
        self.in1 = nn.InstanceNorm1d(hidden_channels)
        self.gat1 = pyg.nn.conv.GeneralConv(hidden_channels, hidden_channels, attention=True)
        self.in2 = nn.InstanceNorm1d(hidden_channels)
        self.gat2 = pyg.nn.conv.GeneralConv(hidden_channels, hidden_channels, attention=True)
        self.in3 = nn.InstanceNorm1d(hidden_channels)
        self.gat3 = pyg.nn.conv.GeneralConv(hidden_channels, hidden_channels, attention=True)
        
        self.decoder = nn.Sequential()
        for _ in range(2):
            self.decoder.append(nn.InstanceNorm1d(hidden_channels))
            self.decoder.append(nn.Conv1d(hidden_channels, hidden_channels, ker_size, padding=(ker_size-1)//2))
            self.decoder.append(nn.ReLU())

        self.decoder.append(nn.Conv1d(hidden_channels, out_channels, ker_size, padding=(ker_size-1)//2))
        self.decoder.append(nn.ReLU())

    def forward(self, x, g):
        x = self.encoder(x)
        x = self.in1(x)
        x = self.gat1(x.T, g)
        x = F.relu(x.T)
        x = self.in2(x)
        x = self.gat2(x.T, g)
        x = F.relu(x.T)
        x = self.in3(x)
        x = self.gat3(x.T, g)
        x = F.relu(x.T)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):

        epi, cage, g, e_a = batch

        pred_cage = self(epi.squeeze(0), g.squeeze())
        
        loss = F.mse_loss(pred_cage, cage)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        with torch.no_grad():
            
            epi, cage, g, e_a = batch

            pred_cage = self(epi.squeeze(0), g.squeeze())
            
            loss = F.mse_loss(pred_cage, cage)
            
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            
        return loss    

    def test_step(self, batch, batch_idx):
        
        with torch.no_grad():
            
            epi, cage, g, e_a = batch

            pred_cage = self(epi.squeeze(0), g.squeeze())
            
            loss = F.mse_loss(pred_cage, cage)
            
            self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            
        return loss    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer