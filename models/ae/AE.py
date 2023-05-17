import importlib, os
from collections import namedtuple
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim as toptim
from torchvision.utils import make_grid
from torchvision import transforms

from models.cnn.CNN import BasicConvNet, DeConvolution_layer, Convolution_layer
from models.mlp.MLP import MultiLayerPerceptron

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

conv_configure= namedtuple("conv_config", ["model", 
                                           "in_channels", "out_channels", 
                                           "k", "s", "p", 
                                           "normalize", "activation", "pooling"])

def get_grid(tensor, image_shape):
        
    if len(tensor.shape) == 2:
        tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
    
    return make_grid(tensor, normalize=True)


class MLPAE(nn.Module):
    def __init__(self, 
                 image_channel,
                 image_size):
        super().__init__()
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.MSELoss()
        
        
        image_features = image_channel * image_size * image_size
        self.encoder = MultiLayerPerceptron(hidden_activation=nn.LeakyReLU(0.02),
                                            final_activation=nn.LeakyReLU(0.02),
                                            features=[image_features,  256, 64, 1])
        
        self.decoder = MultiLayerPerceptron(hidden_activation=nn.LeakyReLU(0.02),
                                            final_activation=nn.Sigmoid(),
                                            features=[1, 64, 256, image_features])
    

    def forward(self, x):
        return self.decoder(self.encoder(x.view(x.size(0), -1)))
    
    
    def get_loss(self, batch, epoch):
        x, _ = batch
        x_hat = self(x.requires_grad_(True))
        
        loss = self.criterion(x.view(x.size(0), -1), x_hat)
        return loss
    
    
class ConvAE(nn.Module):
    def __init__(self, 
                 image_channel,
                 image_size):
        super().__init__()
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.MSELoss()
        
        self.encoder = BasicConvNet(conv_configure(model=Convolution_layer,
                                                   in_channels=[image_channel, 64, 128, 256, 512],
                                                   out_channels=[64, 128, 256, 512, 512],
                                                   k=[4, 4, 4, 4, 2],
                                                   s=[2 for _ in range(5)],
                                                   p=[1, 1, 1, 1, 0],
                                                   normalize=[nn.BatchNorm2d(64),nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(512), nn.BatchNorm2d(512)],
                                                   activation=[nn.LeakyReLU() for _ in range(5)],
                                                   pooling=[None for _ in range(5)]))
        
        self.decoder = BasicConvNet(conv_configure(model=DeConvolution_layer,
                                                   in_channels=[512, 512, 256, 128, 64],
                                                   out_channels=[512, 256, 128, 64, image_channel],
                                                   k=[4, 4, 4, 4, 4],
                                                   s=[2 for _ in range(5)],
                                                   p=[1, 1, 1, 1, 1],
                                                   normalize=[nn.BatchNorm2d(512), nn.BatchNorm2d(256), nn.BatchNorm2d(128), nn.BatchNorm2d(64), nn.BatchNorm2d(image_channel)],
                                                   activation=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Sigmoid()],
                                                   pooling=[None for _ in range(5)]),
                                    output_shape="image",
                                    image_shape=self.image_shape)
    
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    
    def get_loss(self, batch, epoch):
        x, _ = batch
        x_hat = self(x.requires_grad_(True))
        
        loss = self.criterion(x, x_hat)
        return loss
        

class Unet(nn.Module):
    def __init__(self, 
                 image_channel,
                 image_size):
        super().__init__()
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.MSELoss()
        
        self.encoder = BasicConvNet(conv_configure(model=Convolution_layer,
                                                   in_channels=[image_channel, 64, 128, 256, 512],
                                                   out_channels=[64, 128, 256, 512, 512],
                                                   k=[4, 4, 4, 4, 2],
                                                   s=[2 for _ in range(5)],
                                                   p=[1, 1, 1, 1, 0],
                                                   normalize=[nn.BatchNorm2d(64),nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(512), nn.BatchNorm2d(512)],
                                                   activation=[nn.LeakyReLU(0.02) for _ in range(5)],
                                                   pooling=[None for _ in range(5)]))
        
        self.decoder = BasicConvNet(conv_configure(model=DeConvolution_layer,
                                                   in_channels=[1024, 1024, 512, 256, 128],
                                                   out_channels=[512, 256, 128, 64, image_channel],
                                                   k=[4 for _ in range(5)],
                                                   s=[2 for _ in range(5)],
                                                   p=[1 for _ in range(5)],
                                                   normalize=[nn.BatchNorm2d(512), nn.BatchNorm2d(256), nn.BatchNorm2d(128), nn.BatchNorm2d(64), nn.BatchNorm2d(image_channel)],
                                                   activation=[nn.LeakyReLU(0.02), nn.LeakyReLU(0.02), nn.LeakyReLU(0.02), nn.LeakyReLU(0.02), nn.Sigmoid()],
                                                   pooling=[None for _ in range(5)]),
                                    output_shape="image",
                                    image_shape=self.image_shape)
        
        
    def unet_layer(self, x):
        encoder_out = []
        
        for idx, layer in enumerate(self.encoder):
            if idx == 0:
                _out = layer(x)
            else:
                _out = layer(encoder_out[idx -1])
            encoder_out.append(_out)
            
        encoder_out.reverse()
        
        decoder_out = []
        for idx, layer in enumerate(self.decoder):
            if idx == 0:
                _out = layer(torch.cat((encoder_out[idx], encoder_out[idx]), 1))
            else:
                _out = layer(torch.cat((decoder_out[idx -1], encoder_out[idx]), 1))
            decoder_out.append(_out)
            
        return decoder_out[-1]

    def forward(self, x):
        return self.unet_layer(x)
    
    
    def get_loss(self, batch, epoch):
        x, _ = batch
        x_hat = self(x.requires_grad_(True))
        
        loss = self.criterion(x, x_hat)
        return loss
    
    
class LitAE(pl.LightningModule):
    def __init__(self,
                 optim_name: str,
                 lr,
                 model_name,
                 model_args):
        super().__init__()
        self.lr = lr
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
        
    def configure_optimizers(self):
        params = list()
        
        params.extend(list(self.model.encoder.parameters()))
        params.extend(list(self.model.decoder.parameters()))
        
        optim = self.optimizer(params, self.lr)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1)
        
        return [optim], [lr_scheduler]
       
       
    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        # return super().on_train_batch_start(batch, batch_idx)
        pass
    
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        
        self.logger.experiment.add_image("x_hat", get_grid(self.model(batch[0]), self.model.image_shape), self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        pass
    
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
        pass
    
    
    def test_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_test_batch_start(batch, batch_idx, dataloader_idx)
        pass
    
    
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)
        pass
    
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().predict_step(batch, batch_idx, dataloader_idx)
        pass