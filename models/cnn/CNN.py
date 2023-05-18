import importlib
from collections import namedtuple

import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as toptim
import torch.nn.functional as F

from models.mlp.MLP import MultiLayerPerceptron

conv_configure= namedtuple("conv_config", ["model", 
                                           "in_channels", "out_channels", 
                                           "k", "s", "p", 
                                           "normalize", "activation", "pooling"])


class DeConvolution_layer(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, normalize=None, activation=None, pooling=None) -> None:
        super().__init__()
        layer = [
            nn.ConvTranspose2d(in_channels, out_channels, k, s, p)
        ]
        if normalize is not None:
            layer.append(normalize)
        
        if activation is not None:
            layer.append(activation)
        
        if pooling is not None:
            layer.append(pooling)
        
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layer(x)
    
    
class Convolution_layer(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, normalize=None, activation=None, pooling=None):
        super().__init__()
        layer = [
            nn.Conv2d(in_channels, out_channels, k, s, p)
        ]
        if normalize is not None:
            layer.append(normalize)
        
        if activation is not None:
            layer.append(activation)
        
        if pooling is not None:
            layer.append(pooling)
        
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.layer(x)
   

class BasicConvNet(nn.Module):
    def __init__(self, conv_config = None, output_shape = None, image_shape = None):
        super().__init__()
        self.image_shape = image_shape
        self.output_shape = output_shape
        
        layers = []
        for idx in range(len(conv_config.in_channels)):
            layers.append(conv_config.model(conv_config.in_channels[idx], 
                                            conv_config.out_channels[idx], 
                                            conv_config.k[idx], 
                                            conv_config.s[idx], 
                                            conv_config.p[idx],
                                            conv_config.normalize[idx],
                                            conv_config.activation[idx],
                                            conv_config.pooling[idx]))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.output_shape == "image":
            out = self.layers(x).view(x.size(0), self.image_shape[0], self.image_shape[1], self.image_shape[2])
        else:
            out = self.layers(x)
        return out


'''
LeNet5
Input 1x32x32
Output out_features
'''
class LeNet5(nn.Module):
    def __init__(self,
                 image_channel,
                 image_size,
                 out_features):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        self.conv_layers = BasicConvNet(conv_configure(model=Convolution_layer,
                                                       in_channels=[image_channel, 6],
                                                       out_channels=[6, 16],
                                                       k=[5, 5],
                                                       s=[1, 1],
                                                       p=[0, 0],
                                                       normalize=[None, None],
                                                       activation=[nn.ReLU(), nn.ReLU()],
                                                       pooling=[nn.AvgPool2d(2), nn.AvgPool2d(2)]))
        
        self.mlp_layers = MultiLayerPerceptron(hidden_activation=nn.ReLU(),
                                               final_activation=nn.Softmax(1),
                                               features=[16*5*5, 120, 84, out_features])
        
        
    def forward(self, x):
        return self.mlp_layers(self.conv_layers(x).view(x.size(0),-1))
        
        
    def get_loss(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

'''
AlexNet
Input 1x277x277
Output 10
'''
class AlexNet(nn.Module):
    def __init__(self,
                 image_channel,
                 image_size,
                 out_features):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
                
        self.conv_layers = BasicConvNet(conv_configure(model=Convolution_layer,
                                                       in_channels=[image_size, 96, 256, 384, 384],
                                                       out_channels=[96, 256, 384, 256],
                                                       k=[11, 5, 3, 3, 3],
                                                       s=[4, 1, 1, 1, 1],
                                                       p=[0, 2, 1, 1, 1],
                                                       normalize=[None for _ in range(5)],
                                                       activation=[nn.ReLU(), nn.ReLU()],
                                                       pooling=[nn.MaxPool2d(3, 2, 0), nn.MaxPool2d(3, 2, 0), None, None, nn.MaxPool2d(3, 2, 0)]))
        
        self.mlp_layers = MultiLayerPerceptron(hidden_activation=nn.ReLU(),
                                               final_activation=nn.Softmax(1),
                                               features=[256*6*6, 4096, 4096, out_features])
    
    def forward(self, x):
        return self.mlp_layers(self.conv_layers(x).view(x.size(0),-1))
       
           
    def get_loss(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss
    
    
'''
VGGNet
Input 1x224x224
Output 10
'''
class VGGNet(nn.Module):
    def get_vgg_config(self, vgg_layers=None):
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
                 image_channel,
                 image_size,
                 out_features = None,
                 hidden_activation = nn.ReLU(),
                 final_activation = nn.Softmax(1),
                 vgg_layers = None,
                 features = None,):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        channels = self.get_vgg_config(vgg_layers) # [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        if features is None:
            features = [512*7*7, 4096, 4096, out_features]
        
        conv_layers = []
        for idx in range(len(channels)):
            pooling, normalize = None, None
            
            if channels[idx] in ("M", "A"):
                continue
            elif channels[idx +1] == "M":
                pooling = nn.MaxPool2d(2, 2)
            elif channels[idx +1] == "A":
                pooling = nn.AvgPool2d(2, 2)
                
            _in_channel = image_channel
            _out_channel = channels[idx]
            k, s, p = 3, 1, 1
            
            normalize = nn.BatchNorm2d(_out_channel)
                
            conv_layers.append(Convolution_layer(_in_channel, _out_channel, k, s, p, normalize, nn.ReLU(), pooling))
            image_channel = _out_channel
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.mlp_layers = MultiLayerPerceptron(hidden_activation=hidden_activation,
                                               final_activation=final_activation,
                                               features=features)
    
    def forward(self, x):
        return self.mlp_layers(self.conv_layers(x).view(x.size(0),-1))
       
           
    def get_loss(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss
       

'''
ResNet
Input 1x224x224
Output 10
''' 
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self,
                 in_channels,
                 plans,
                 stride = 1,
                 downsample = False,
                 norm_layer = None):
        super().__init__()
        out_channels = plans * self.expansion
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        layer = [
            Convolution_layer(in_channels, plans, 3, stride, 1, norm_layer(plans), nn.ReLU()),
            Convolution_layer(plans, out_channels, 3, 1, 1, norm_layer(out_channels))
        ]
        
        self.downsample = None
        if downsample:
            self.downsample = Convolution_layer(in_channels, out_channels, 1, stride, 0, norm_layer(out_channels))
            
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
            
        return F.relu(identity + self.layer(x))
        
        
class Bottleneck(nn.Module):
    expansion = 4
     
    def __init__(self, 
                 in_channels,
                 planes,
                 stride = 1,
                 downsample = None,
                 norm_layer = None):
        super().__init__()
        out_channels =  planes * self.expansion
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        layer = [
            Convolution_layer(in_channels, planes, 1, 1, 0, norm_layer(planes), nn.ReLU()),
            Convolution_layer(planes, planes, 3, stride, 1, norm_layer(planes), nn.ReLU()),
            Convolution_layer(planes, out_channels, 1, 1, 0, norm_layer(out_channels))
        ]
        
        self.downsample = None
        if downsample:
            self.downsample = Convolution_layer(in_channels, out_channels, 1, stride, 0, norm_layer(out_channels))
            
        self.layer = nn.Sequential(*layer)
        
    def forward(self, x):
        if self.downsample is not None:
            identitiy = self.downsample(x)
        else:
            identitiy = x
            
        return F.relu(identitiy + self.layer(x))
    
    
class ResNet(nn.Module):
    def get_resnet_config(self, res_layers):

        resnet_config = namedtuple("resnet_config", ["block", "n_blocks", "channels"])

        if res_layers == 34:
            config = resnet_config(
                block=BasicBlock,
                n_blocks=[3, 4, 6, 3],
                channels=[64, 128, 256, 512])

        elif res_layers == 50:
            config = resnet_config(
                block=Bottleneck,
                n_blocks=[3, 4, 6, 3],
                channels=[64, 128, 256, 512])
        elif res_layers == 101:
            config = resnet_config(
                block=Bottleneck,
                n_blocks=[3, 4, 23, 3],
                channels=[64, 128, 256, 512])
        elif res_layers == 152:
            config = resnet_config(
                block=Bottleneck,
                n_blocks=[3, 8, 36, 3],
                channels=[64, 128, 256, 512])
        else: # layers == 18 or enter wrong number
            res_layers = 18
            config = resnet_config(
                block=BasicBlock,
                n_blocks=[2, 2, 2, 2],
                channels=[64, 128, 256, 512])
        
        return config
    
    
    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []
        
        if block.expansion * channels != self.in_channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for _ in range(1, n_blocks):
            layers.append(block(block.expansion*channels, channels))
        
        self.in_channels = block.expansion * channels
        
        return nn.Sequential(*layers)
    
    
    def __init__(self,
                 image_channel,
                 image_size,
                 out_features,
                 hidden_activation = nn.ReLU(),
                 final_activation = nn.Softmax(1),
                 res_layers = 18,
                 features = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        block, n_blocks, channels = self.get_resnet_config(res_layers)
        self.in_channels = channels[0]
        
        if features is None:
            features = [512 * block.expansion, out_features]
        
        conv_layers = [Convolution_layer(image_channel, 
                                         self.in_channels, 
                                         7, 2, 3, 
                                         nn.BatchNorm2d(self.in_channels), 
                                         hidden_activation, 
                                         nn.MaxPool2d(3, 2, 1))]
        
        for idx in range(len(channels)):
            stride = 1 if idx == 0 else 2

            conv_layers.append(self.get_resnet_layer(block, n_blocks[idx], channels[idx], stride))
        
        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.mlp_layers = MultiLayerPerceptron(hidden_activation=hidden_activation,
                                               final_activation=final_activation,
                                               features=features)
        
        
    def forward(self, x):
        return self.mlp_layers(self.conv_layers(x).view(x.size(0), -1))
       
           
    def get_loss(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss


class LitCNN(pl.LightningModule):
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
        loss = self.model.get_loss(batch)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch)
        self.log("val_loss", loss, sync_dist=True)
        return loss
        
    
    def test_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch)
        self.log("test_loss", loss, sync_dist=True)
        return loss
    
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), self.lr)
    