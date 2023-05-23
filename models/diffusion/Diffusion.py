import importlib, os
from collections import namedtuple
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as toptim
from torchvision.utils import make_grid
from torchvision import transforms

from models.cnn.CNN import BasicConvNet, DeConvolution_layer, Convolution_layer
from models.mlp.MLP import MultiLayerPerceptron
from run import get_obj_from_str


def get_grid(tensor, image_shape):
        
    if len(tensor.shape) == 2:
        tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
    
    return make_grid(tensor, normalize=True)

    
class DDPM(nn.Module):
    def __init__(self,
                 eps_model_name:str,
                 eps_model_args:dict,
                 image_channel=3,
                 image_size=32,
                 n_steps=1_000,
                 n_samples=16):
        super().__init__()
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.MSELoss()
        
        eps_model_args.update({"image_channel": image_channel, "image_size": image_size})
        self.eps_model = get_obj_from_str(eps_model_name)(**eps_model_args)
        
        self.n_steps = n_steps
        self.n_samples = n_samples
        
        self.beta = torch.linspace(0.0001, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.sigma2 = self.beta
            
    def gather(self, consts, t):
        c = torch.gather(consts, -1, t)
        # c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)
    
    
    def q_xt_x0(self, x0, t):
        mean = torch.sqrt(self.gather(self.alpha_bar, t)) * x0
        var = 1 - self.gather(self.alpha_bar, t)
        return mean, var
    
    
    def q_sample(self, x0, t, eps):            
        mean, var = self.q_xt_x0(x0, t)
        
        return mean + (torch.sqrt(var)) * eps
    
    
    def p_sample(self, xt, t):
        eps_theta = self.eps_model(xt, t)
        
        alpha = self.gather(self.alpha, t)
        beta = 1 - alpha
        
        alpha_bar = self.gather(self.alpha_bar, t)
        beta_bar = 1 - alpha_bar
        
        eps_coef = torch.div(beta, torch.sqrt(beta_bar))
        
        mean = torch.div(1, torch.sqrt(alpha)) * (xt - eps_coef * eps_theta)
        var = self.gather(self.sigma2, t)
        
        eps = torch.randn_like(xt, device=xt.device)
        
        return mean + torch.sqrt(var) * eps
    
    
    def forward(self):
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long, device=x.device))
        
        return x
    
    
    def get_loss(self, batch, epoch):
        x0, _ = batch
        self.beta = self.beta.to(x0.device)
        self.alpha_bar = self.alpha_bar.to(x0.device)
        t = torch.randint(0, self.n_steps, (x0.size(0), ), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0, device=x0.device)
        
        xt = self.q_sample(x0, t, noise)
        eps_theta = self.eps_model(xt, t)
        
        loss = self.criterion(noise, eps_theta)

        return loss


class DDIM(nn.Module):
    def __init__(self,
                 image_channel=3,
                 image_size=32):
        super().__init__()
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.MSELoss()
        
    
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
    
    
    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        
        self.logger.experiment.add_image("x_hat", get_grid(self.model(), self.model.image_shape), self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss