import importlib

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as toptim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    
 
class LitMLP(pl.LightningModule):
    def __init__(self, 
                 lr,
                 model_name,
                 model_args):
        super().__init__()
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
    def forward(self, x):
        return self.model(x)
    
    def get_loss(self, batch, log_string):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(log_string, loss, sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, "train_loss")
         
    def validation_step(self, batch, batch_idx):
        self.get_loss(batch, "val_loss")
    
    def test_step(self, batch, batch_idx):
        return self.get_loss(batch, "test_loss")
    
    def configure_optimizers(self):
        return toptim.Adam(self.parameters(), self.lr)