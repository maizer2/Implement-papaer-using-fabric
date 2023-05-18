import importlib, os
from collections import namedtuple
from typing import Any, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as toptim
from torchvision.utils import make_grid
from torchvision import transforms

from models.cnn.CNN import BasicConvNet, DeConvolution_layer, Convolution_layer
from models.mlp.MLP import MultiLayerPerceptron
from run import get_obj_from_str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gather(consts, t):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)
    
    
class DDPM(nn.Module):
    def __init__(self,
                 image_channel,
                 image_size,
                 n_steps,
                 eps_model_name:str,
                 eps_model_args:dict):
        super().__init__()
        
        self.image_channel = image_channel
        self.image_size = image_size
        
        self.beta = torch.linspace(0.0001, 0.02, n_steps, device=device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        
        eps_model_args.update({"image_channel": image_channel, "image_size": image_size})
        self.eps_model = get_obj_from_str(eps_model_name)(**eps_model_args)
        
        
    def q_xt_x0(self, x0, t):
        mean = torch.sqrt(gather(self.alpha_bar, t)) * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean, var
    
    
    def q_sample(self, x0, t, eps = None):
        if eps is None:
            eps = torch.randn_like(x0)
        
        mean, var = self.q_xt_x0(x0, t)
        
        return mean + (torch.sqrt(var)) * eps
    
    
    def p_sample(self, xt, t):
        eps_theta = self.eps_model(xt, t)
        alpha = gather(self.alpha, t)
        alpha_bar = gather(self.alpha_bar, t)
        beta = 1 - alpha
        
        eps_coef = torch.div(beta, torch.sqrt(beta))
        
        mean = torch.div(1, torch.sqrt(alpha)) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        
        eps = torch.randn_like(xt, device=device)
        
        return mean + torch.sqrt(var) * eps
    
    
    def loss(self, x0, noise = None):
        t = torch.randint(0, self.n_steps, (x0.size(0), ), device=device, dtype=torch.long)
        
        if noise is None:
            noise = torch.rand_like(x0)
        
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        
        return F.mse_loss(noise, eps_theta)
        
    
    def forward(self, x):
        pass
    
    
    def get_loss(self, batch,):
        pass
    

class DDIM(nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, x):
        pass
    
    
    def get_loss(self, batch):
        pass
    

class LitDiffusion(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 optim_name: str,
                 model_name: str,
                 model_args: tuple) -> None:
        super().__init__()
        self.lr = lr
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
    
    
    def configure_optimizers(self):
        optim = self.optimizer(self.model.parameters(), self.lr)
        return optim
    
    
    def training_step(self, *args: Any, **kwargs: Any):
        return super().training_step(*args, **kwargs)
    
    
    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)
    
    
    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)