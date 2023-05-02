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
        
        self.layer = nn.Sequential(layer)
    
    def forward(self, x):
        return self.layer(x)
    
class LitMultiLayerPerceptron(pl.LightningDataModule):
    def __init__(self):
        
        features = [32, 16, 8, 4, 2, 1]
        layers = []
        
        for feature in features:
            if feature != 2:
                layers.append(SinglePerceptron(feature, feature/2))
            else:
                layers.append(SinglePerceptron(feature, feature/2, nn.Softmax(1)))
                break
            
        self.layers = nn.Sequential(*layers)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        criterion = nn.CrossEntropyLoss()
        
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        
        loss = criterion(y_hat, y)
        return loss
        
    def configure_optimizers(self, lr):
        optimizer = toptim.Adam(self.parameters(), lr)
        return optimizer