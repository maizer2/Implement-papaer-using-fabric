import importlib

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as toptim

'''
MLP
Input 1x32x32
Output 10
'''

class SinglePerceptron(nn.Module):
    def __init__(self, in_features, out_features, dropout:bool = True, normalize = None, activation: nn = nn.ReLU()):
        super().__init__()
        layer = [
            nn.Linear(in_features, out_features)
            ]
        
        if dropout:
            layer.append(nn.Dropout())
        
        if normalize is not None:
            layer.append(normalize(out_features, 0.8))
            
        if activation is not None:
            layer.append(activation)
        
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.layer(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, 
                 in_features = None,
                 out_features = None,
                 normalize = None,
                 hidden_activation = nn.ReLU(),
                 final_activation = nn.Softmax(1),
                 features = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        if features is None:
            features = [in_features, 512, 256, 128, 64, 32, 16, out_features]
            
        layers = []
        for idx in range(len(features) -1):
            if features[idx] != features[-2]:
                activation=hidden_activation
            else:
                activation=final_activation
            layers.append(SinglePerceptron(features[idx], features[idx +1], normalize=normalize, activation=activation))
            
        self.layers = nn.Sequential(*layers)
        
    
    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))
    
    
    def get_loss(self, batch, epoch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss
 
 
class LitMLP(pl.LightningModule):
    def __init__(self, 
                 lr,
                 optim_name,
                 model_name,
                 model_args):
        super().__init__()
        self.lr = lr
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch, self.device)
        self.log("train_loss", loss, sync_dist=True)
        return loss
         
         
    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch, self.device)
        self.log( "val_loss", loss, sync_dist=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch, self.device)
        self.log("test_loss", loss, sync_dist=True)
        return loss
    
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), self.lr)