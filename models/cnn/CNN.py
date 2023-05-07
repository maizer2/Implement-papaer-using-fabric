import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as toptim

from models.mlp.MLP import LitMultiLayerPerceptron

class Convolution_layer(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, pooling=None):
        super().__init__()
        layer = [
            nn.Conv2d(in_channels, out_channels, k, s, p),
            nn.ReLU()
        ]
        
        if pooling is not None:
            layer.append(pooling)
        
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layer(x)
   
    
'''
LeNet5
Input 1x32x32
Output 10
'''
class LeNet5(nn.Module):
    def __init__(self,
                 in_channels = None,
                 out_features = None,
                 channels = None,
                 features = None):
        super().__init__()
        
        if channels is None:
            channels = [in_channels, 6, 16]
        if features is None:
            features = [16*5*5, 120, 84, out_features]
        
        conv_layers = []
        for idx in range(len(channels) -1):
            k, s, p = 5, 1, 0
            pooling = nn.AvgPool2d(2)
            conv_layers.append(Convolution_layer(channels[idx], channels[idx+1], k, s, p, pooling))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.mlp_layers = LitMultiLayerPerceptron(features=features).layers
        
    def forward(self, x):
        return self.mlp_layers(self.conv_layers(x).view(x.size(0),-1))
        

'''
AlexNet
Input 1x277x277
Output 10
'''
class LitAlexNet(nn.Module):
    def __init__(self,
                 in_channels = None,
                 out_features = None,
                 channels = None,
                 features = None):
        super().__init__()
        
        if channels is None:
            channels = [in_channels, 96, 256, 384, 384, 256]
        if features is None:
            features = [256*6*6, 4096, 4096, out_features]
        
        conv_layers = []
        for idx in range(len(channels) -1):
            k, s, p = 3, 1, 1
            pooling = nn.MaxPool2d(3, 2, 0)
            
            if idx == 0:
                k, s, p = 11, 4, 0
            elif idx == 1:
                k, s, p = 5, 1, 2
            elif idx in (2, 3):
                pooling=None    
                
            conv_layers.append(Convolution_layer(channels[idx], channels[idx+1], k, s, p, pooling))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.mlp_layers = LitMultiLayerPerceptron(features=features).layers
    
    def forward(self, x):
        return self.mlp_layers(self.conv_layers(x).view(x.size(0),-1))
       
           
'''
VGGNet
Input 1x224x224
Output 10
'''
class LitVGGNet(nn.Module):
    def get_vgg_config(vgg_layers):

        if vgg_layers == 13:
            vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        elif vgg_layers == 16:
            vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        elif vgg_layers == 19:
            vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

        else: # layers == 11 or enter wrong number
            vgg_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        return vgg_config
    
    def __init__(self,
                 in_channels = None,
                 out_features = None,
                 vgg_layers = None,
                 channels = None,
                 features = None):
        super().__init__()
        
        if channels is None:
            channels = [in_channels, 96, 256, 384, 384, 256]
        if features is None:
            features = [256*6*6, 4096, 4096, out_features]
        
        conv_layers = []
        for idx in range(len(channels) -1):
            k, s, p = 3, 1, 1
            pooling = nn.MaxPool2d(3, 2, 0)
            
            if idx == 0:
                k, s, p = 11, 4, 0
            elif idx == 1:
                k, s, p = 5, 1, 2
            elif idx in (2, 3):
                pooling=None    
                
            conv_layers.append(Convolution_layer(channels[idx], channels[idx+1], k, s, p, pooling))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.mlp_layers = LitMultiLayerPerceptron(features=features).layers
    
    def get_loss(self, x):
        return self.mlp_layers(self.conv_layers(x).view(x.size(0),-1))
       
           
class LitCNN(pl.LightningModule):
    def __init__(self, 
                 lr,
                 model_args,
                 criterion = nn.CrossEntropyLoss()):
        super().__init__()
        self.lr = lr
        self.criterion = criterion
        self.model = LeNet5(**model_args)
        
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
    