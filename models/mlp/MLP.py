import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as toptim

class SinglePerceptron(nn.Module):
    def __init__(self, in_features, out_features, dropout:bool = True, activation: nn = nn.ReLU()):
        super().__init__()
        layer = [
            nn.Linear(in_features, out_features)
            ]
        
        if dropout:
            layer.append(nn.Dropout())
        
        layer.append(activation)
        
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.layer(x)
    
class LitMultiLayerPerceptron(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.criterion = nn.CrossEntropyLoss()
        
        features = [1024, 512, 256, 128, 64, 32, 16, 10]
        layers = []
        
        for feature in features:
            if feature != 16:
                layers.append(SinglePerceptron(feature, feature//2))
            else:
                layers.append(SinglePerceptron(feature, features[-1], activation=nn.Softmax(1)))
                break
            
        self.layers = nn.Sequential(*layers)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x.view(x.size(0), -1))
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
    
        return loss
         
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x.view(x.size(0), -1))
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x.view(x.size(0), -1))
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = toptim.Adam(self.parameters(), self.lr)
        return optimizer