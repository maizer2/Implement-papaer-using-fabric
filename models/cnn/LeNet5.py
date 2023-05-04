import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as toptim

from models.mlp.MLP import LitMultiLayerPerceptron

'''
LeNet5
Input 1x32x32
Output 10
'''


class Convolution_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layer = [
            nn.Conv2d(in_channels, out_channels, 5, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2)
        ]
        
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layer(x)
    

class LitLeNet5(pl.LightningModule):
    def __init__(self,
                 lr,
                 in_channels = None,
                 out_features = None,
                 channels = None,
                 features = None):
        super().__init__()
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        
        if channels is None:
            channels = [in_channels, 6, 16]
        if features is None:
            features = [16*5*5, 120, 84, out_features]
        
        conv_layers = []
        for idx in range(len(channels) -1):
            conv_layers.append(Convolution_layer(channels[idx], channels[idx+1]))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.mlp_layers = LitMultiLayerPerceptron(features=features).layers
        
    
    def get_loss(self, batch, batch_idx, log_string):
        x, y = batch
        y_hat = self.mlp_layers(self.conv_layers(x).view(x.size(0),-1))
        loss = self.criterion(y_hat, y)
        self.log(log_string, loss, sync_dist=True)
        return loss
        
        
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx, "train_loss")
    
    
    def validation_step(self, batch, batch_idx):
        self.get_loss(batch, batch_idx, "val_loss")
        
    
    def test_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx, "test_loss")
    
    
    def configure_optimizers(self):
        return toptim.Adam(self.parameters(), self.lr)