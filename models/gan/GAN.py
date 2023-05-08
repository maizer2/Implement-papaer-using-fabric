import importlib
from collections import namedtuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as toptim
import torch.nn.functional as F

from models.mlp.MLP import LitMultiLayerPerceptron


class VanilaGenerator(nn.Module):
    def __init__(self,
                 latent_dim,
                 out_features,
                 features = None):
        super().__init__()
        
        if features is None:
            features = [latent_dim, 256, 512, 1024, out_features]
        
        self.mlp_layers = LitMultiLayerPerceptron(features=features,
                                                  hidden_activation=nn.LeakyReLU(0.02), 
                                                  final_activation=nn.Tanh()).layers        
        
    def forward(self, z):
        return self.mlp_layers(z)

class VanilaDiscriminator(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 features = None):
        super().__init__()
        
        if features is None:
            features = [in_features, 256, 512, 1024, out_features]
        
        self.mlp_layers = LitMultiLayerPerceptron(features=features,
                                                  hidden_activation=nn.ReLU(),
                                                  final_activation=nn.Sigmoid()).layers
        
    def forward(self, x):
        return self.mlp_layers(x)
    
class LitGAN(pl.LightningModule):
    def __init__(self,
                 lr,
                 latent_dim,
                 G_name, G_args,
                 D_name, D_args,
                 criterion = nn.BCELoss()):
        super().__init__()
        self.lr = lr
        self.latent_dim = latent_dim
        self.criterion = criterion
        self.G = getattr(importlib.import_module(__name__), G_name)(**G_args)
        self.D = getattr(importlib.import_module(__name__), D_name)(**D_args)
        
    def get_loss(self, batch, log_g_string, log_d_string):
        optim_g, optim_d = self.optimizers()
        
        # Training Generator
        self.toggle_optimizer(optim_g)
        
        real, _ = batch
        z = torch.randn(self.latent_dim)
        fake = self.G(z)
        
        label_real = torch.ones_like(fake)
        label_fake = torch.zeros_like(fake)
        
        pred_real = self.D(real)
        pred_fake = self.D(fake)
        
        loss_g = self.criterion(pred_fake, label_real)
        self.manual_backward(loss_g)
        optim_g.step()
        optim_g.zero_grad()
        self.untoggle_optimizer(optim_g)
        
        # Training Discriminator
        self.toggle_optimizer(optim_d)
        z = torch.randn(self.latent_dim)
        fake = self.G(z).detach()
        
        pred_real = self.D(real)
        pred_fake = self.D(fake)
        
        loss_d = ( self.criterion(pred_real, label_real) + self.criterion(pred_fake, label_fake) ) / 2
        self.manual_backward(loss_d)
        optim_d.step()
        optim_d.zero_grad()
        self.untoggle_optimizer(optim_d)
        
        self.log(log_g_string, loss_g, prog_bar=True)
        self.log(log_d_string, loss_d, prog_bar=True)
        
        
    def training_step(self, batch, batch_idx):
        self.get_loss(batch, "loss_g", "loss_d")
    
    
    def validation_step(self, batch, batch_idx):
        self.get_loss(batch, "loss_g", "loss_d")
        
    
    def test_step(self, batch, batch_idx):
        self.get_loss(batch, "loss_g", "loss_d")
    
    
    def configure_optimizers(self):
        optim_g = toptim.Adam(self.G.parameters(), self.lr)
        optim_d = toptim.Adam(self.D.parameters(), self.lr)
        return [optim_g, optim_d], []
    