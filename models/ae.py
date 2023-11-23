import importlib
import math
import os
from collections import namedtuple
from typing import Any, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as toptim
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision import transforms
from torchvision.utils import make_grid

from models.cnn.CNN import BasicConvNet, Convolution_layer, DeConvolution_layer
from models.mlp.MLP import MultiLayerPerceptron

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
    
    
class Unet_diff(nn.Module):
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
        
        
    class TimeEmbedding(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            self.n_channels = n_channels
            
            self.linear1 = nn.Linear(self.n_channels // 4, self.n_channels)
            self.activation = Unet_diff.Swish()
            self.linear2 = nn.Linear(self.n_channels, self.n_channels)
        
        
        def forward(self, t):
            half_dim = self.n_channels // 8
            emb =  math.log(10_000) / (half_dim  - 1)
            emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
            emb = t[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), 1)
            
            emb = self.activation(self.linear1(emb))
            emb = self.linear2(emb)
            
            return emb
            
            
    class ResidualBlock(nn.Module):
        def __init__(self,
                     in_channels,
                     out_channels,
                     time_channels,
                     n_groups = 32,
                     dropout = 0.1,):
            super().__init__()
            
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.act1 = Unet_diff.Swish()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
            self.act2 = Unet_diff.Swish()
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            else:
                self.shortcut = nn.Identity()
                
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Unet_diff.Swish()
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, t):
            
            h = self.conv1(self.act1(self.norm1(x)))
            h += self.time_emb(self.time_act(t))[:, :, None, None]
            h = self.conv2(self.dropout(self.act2(self.norm2(h))))
            
            return h + self.shortcut(x)
        
    
    class AttentionBlock(nn.Module):
        def __init__(self,
                     n_channels,
                     n_heads = 1,
                     d_k = None,
                     n_groups = 32):
            super().__init__()
            
            if d_k is None:
                d_k = n_channels
                
            self.norm = nn.GroupNorm(n_groups, n_channels)
            self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
            self.output = nn.Linear(n_heads * d_k, n_channels)
            self.scale = d_k ** -0.5
            
            self. n_heads = n_heads
            self.d_k = d_k
            
        def forward(self, x, t = None):
            _ = t

            batch_size, n_channels, height, width = x.shape
            
            x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
            qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3* self.d_k)
            q, k, v = torch.chunk(qkv, 3, -1)
            
            attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
            attn = attn.softmax(2)
            
            res = torch.einsum('bijh, bjhd->bihd', attn, v)
            res = res.view(batch_size, -1, self.n_heads * self.d_k)
            res = self.output(res)
            
            res += x
            
            return res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        
    
    class DownBlock(nn.Module):
        def __init__(self,
                     in_channels,
                     out_channels,
                     time_channels,
                     has_attn):
            super().__init__()
            self.res = Unet_diff.ResidualBlock(in_channels, out_channels, time_channels)
            
            if has_attn:
                self.attn = Unet_diff.AttentionBlock(out_channels)
            else:
                self.attn = nn.Identity()
                
        def forward(self, x, t):
            x = self.res(x, t)
            x = self.attn(x)
            return x
    
    
    class UpBlock(nn.Module):
        def __init__(self,
                     in_channels,
                     out_channels,
                     time_channels,
                     has_attn):
            super().__init__()
            self.res = Unet_diff.ResidualBlock(in_channels + out_channels, out_channels, time_channels)
            
            if has_attn:
                self.attn = Unet_diff.AttentionBlock(out_channels)
            else:
                self.attn = nn.Identity()
        
        def forward(self, x, t):
            x = self.res(x, t)
            x = self.attn(x)
            return x
    
    
    class MiddleBlock(nn.Module):
        def __init__(self,
                     n_channels,
                     time_channels):
            super().__init__()
            
            self.res1 = Unet_diff.ResidualBlock(n_channels, n_channels, time_channels)
            self.attn = Unet_diff.AttentionBlock(n_channels)
            self.res2 = Unet_diff.ResidualBlock(n_channels, n_channels, time_channels)
            
        def forward(self, x, t):
            x = self.res1(x, t)
            x = self.attn(x)
            x = self.res2(x, t)
            
            return x
    
    
    class Upsample(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            self.conv = nn.ConvTranspose2d(n_channels, n_channels, 4, 2, 1)
            
        def forward(self, x, t):
            _ = t
            return self.conv(x)
    
    
    class Downsample(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            self.conv = nn.Conv2d(n_channels, n_channels, 3, 2, 1)
            
        def forward(self, x, t):
            _ = t
            return self.conv(x)
            
            
    def __init__(self,
                 image_channel,
                 image_size,
                 n_channels = 64,
                 ch_mults = [1, 2, 2, 4],
                 is_attn = [False, False, True, True],
                 n_blocks = 2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_channel = image_channel
        self.image_size = image_size
        
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channel, n_channels, 3, 1, 1)
        self.time_emb = self.TimeEmbedding(n_channels * 4)
        
        down = []
        
        out_channels = in_channels = n_channels
        
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            
            for _ in range(n_blocks):
                down.append(self.DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
                
            if i < n_resolutions - 1:
                down.append(self.Downsample(in_channels))
                
        self.down = nn.ModuleList(down)
        
        self.middle = self.MiddleBlock(out_channels, n_channels * 4)
        
        up = []
        
        in_channels = out_channels
        
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            
            for _ in range(n_blocks):
                up.append(self.UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            
            out_channels = in_channels // ch_mults[i]
            up.append(self.UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            
            if i > 0:
                up.append(self.Upsample(in_channels))
                
        self.up = nn.ModuleList(up)
        
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = self.Swish()
        self.final = nn.Conv2d(in_channels, image_channel, 3, 1, 1)
       
        
    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.image_proj(x)
        h = [x]
        
        for m in self.down:
            x = m(x, t)
            h.append(x)
        
        x = self.middle(x, t)
        
        for m in self.up:
            if isinstance(m, self.Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), 1)
                x = m(x, t)
                
        return self.final(self.act(self.norm(x)))
    
    
    def get_loss(self, batch, epoch):
        x, _ = batch
        x_hat = self(x)
        
        loss = self.criterion(x, x_hat)
        return loss
    
    
class Lit_ae(pl.LightningModule):
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
        loss = self.model.get_loss(batch, self.current_epoch, self.device)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        # return super().on_train_batch_start(batch, batch_idx)
        pass
    
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        
        self.logger.experiment.add_image("x_hat", get_grid(self.model(batch[0]), self.model.image_shape), self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch, self.device)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        pass
    
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
        pass
    
    
    def test_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch, self.device)
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