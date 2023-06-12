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
        
        
    def gather(self, consts, t):
        c = torch.gather(consts, -1, t)
        # c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)
    
    
    def q_xt_x0(self, x0, t, alpha_bar):
        mean = torch.sqrt(self.gather(alpha_bar, t)) * x0
        var = 1 - self.gather(alpha_bar, t)
        return mean, var
    
    
    def q_sample(self, x0, t, eps, alpha_bar):            
        mean, var = self.q_xt_x0(x0, t, alpha_bar)
        
        return mean + (torch.sqrt(var)) * eps
    
    
    def p_sample(self, xt, t, alpha):
        alpha_bar = torch.cumprod(alpha, 0)
        sigma2 = self.beta
        
        eps_theta = self.eps_model(xt, t)
        
        alpha = self.gather(alpha, t)
        beta = 1 - alpha
        
        alpha_bar = self.gather(alpha_bar, t)
        beta_bar = 1 - alpha_bar
        
        eps_coef = torch.div(beta, torch.sqrt(beta_bar))
        
        mean = torch.div(1, torch.sqrt(alpha)) * (xt - eps_coef * eps_theta)
        var = self.gather(sigma2, t)
        
        eps = torch.randn_like(xt, device=xt.device)
        
        return mean + torch.sqrt(var) * eps
    
    
    def forward(self, z):
        self.beta = self.beta.to(z.device)
        alpha = 1. - self.beta
        
        with torch.no_grad():
            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                t = z.new_full((self.n_samples,), t, dtype=torch.long, device=z.device)
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                z = self.p_sample(z, t, alpha)
        
        return z
    
    
    def get_loss(self, batch, epoch):
        
        x0, _ = batch
        self.beta = self.beta.to(x0.device)
        alpha = 1. - self.beta
        alpha_bar = torch.cumprod(alpha, 0)
        
        t = torch.randint(0, self.n_steps, (x0.size(0), ), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0, device=x0.device)
        
        xt = self.q_sample(x0, t, noise, alpha_bar)
        eps_theta = self.eps_model(xt, t)
        loss = self.criterion(noise, eps_theta)

        return loss


class DDIM(nn.Module):
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
        
        
    def get_grid(self, tensor, image_shape):
        
        if len(tensor.shape) == 2:
            tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
        
        return make_grid(tensor, normalize=True)
    
    
    def configure_optimizers(self):
        optim = self.optimizer(self.model.eps_model.parameters(), self.lr)
        return optim
    
    
    def training_step(self, batch, batch_idx):
        # self.model.set_variable_device(self.device)
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        z = torch.randn([self.model.n_samples, self.model.image_shape[0], self.model.image_shape[1], self.model.image_shape[2]],
                        device=self.device)
        
        self.logger.experiment.add_image("x_hat", self.get_grid(self.model(z), self.model.image_shape), self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        # self.model.set_variable_device(self.device)
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        # self.model.set_variable_device(self.device)
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        z = torch.randn([self.model.n_samples, self.model.image_shape[0], self.model.image_shape[1], self.model.image_shape[2]],
                        device=self.device)
        
        x_hat = self.model(z)
        return x_hat
    
    
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        out_path = os.path.join(self.trainer.log_dir, "output_predict")
        os.makedirs(out_path, exist_ok=True)
        
        x_hat_grid = self.get_grid(outputs, outputs.shape)
        x_hat_PIL = transforms.ToPILImage()(x_hat_grid)
        x_hat_PIL.save(os.path.join(out_path, f"{batch_idx}.png"))