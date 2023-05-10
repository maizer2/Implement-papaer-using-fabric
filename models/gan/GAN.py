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
    

class VanilaGAN(nn.Module):
    def __init__(self, 
                 lr, 
                 latent_dim, 
                 G_name, G_args,
                 D_name, D_args,
                 criterion = nn.BCELoss()):
        self.automatic_optimization = False
        self.lr = lr
        self.latent_dim = latent_dim
        self.criterion = criterion
        self.G = getattr(importlib.import_module(__name__), G_name)(**G_args)
        self.D = getattr(importlib.import_module(__name__), D_name)(**D_args)
    
    
    def get_G_loss(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim).cuda()
        fake = self.G(z)
        
        label_real = torch.ones(batch_size, 1).cuda()
        pred_fake = self.D(fake)
        
        loss_g = self.criterion(pred_fake, label_real)
        return loss_g.requires_grad_(True)
        
        
    def get_D_loss(self, batch_size, real):
        real = real.view(batch_size, -1)
        z = torch.randn(batch_size, self.latent_dim).cuda()
        fake = self.G(z).detach()
        
        label_real = torch.ones(batch_size, 1).cuda()
        label_fake = torch.zeros(batch_size, 1).cuda()
        pred_real = self.D(real)
        pred_fake = self.D(fake)
        
        loss_d = ( self.criterion(pred_real, label_real).requires_grad_(True) + self.criterion(pred_fake, label_fake).requires_grad_(True) ) / 2
        return loss_d.requires_grad_(True)
    
    
    
class LitGAN(pl.LightningModule):
    def __init__(self,
                 lr,
                 latent_dim,
                 G_name, G_args,
                 D_name, D_args,
                 criterion = nn.BCELoss()):
        super().__init__()
        
    def get_loss(self, batch, log_g_string, log_d_string):
        optim_g, optim_d = self.optimizers()
        real, _ = batch
        batch_size = real.size(0)
        
        
        # Training Generator
        self.toggle_optimizer(optim_g)
        
        loss_g = self.get_G_loss(batch_size)
        self.log(log_g_string, loss_g, prog_bar=True, prog_dist=True)
        self.manual_backward(loss_g)
        optim_g.step()
        optim_g.zero_grad()
        
        self.untoggle_optimizer(optim_g)
        
        # Training Discriminator
        self.toggle_optimizer(optim_d)
        
        loss_d = self.get_D_loss(batch_size, real)
        self.log(log_d_string, loss_d, prog_bar=True, prog_dist=True)
        self.manual_backward(loss_d)
        optim_d.step()
        optim_d.zero_grad()
        
        self.untoggle_optimizer(optim_d)
        
        
        
    def training_step(self, batch, batch_idx):
        self.get_loss(batch, "loss_g", "loss_d")
    
    
    def validation_step(self, batch, batch_idx):
        self.get_loss(batch, "loss_g", "loss_d")
        
    
    def test_step(self, batch, batch_idx):
        self.get_loss(batch, "loss_g", "loss_d")
    
    
    def configure_optimizers(self):
        optim_g = toptim.Adam(self.G.parameters(), self.lr)
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g, step_size=1)
        optim_d = toptim.Adam(self.D.parameters(), self.lr)
        lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, step_size=1)
        return [optim_g, optim_d], [lr_scheduler_g, lr_scheduler_d]
    
    
    # def predict_step(self, batch):