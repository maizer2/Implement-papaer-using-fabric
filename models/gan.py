import importlib, os
from collections import namedtuple
from typing import Any, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as toptim
from torchvision.utils import make_grid
from torchvision import transforms

from models.ae.CNN import BasicConvNet, DeConvolution_layer, Convolution_layer
from models.mlp.MLP import MultiLayerPerceptron

conv_configure= namedtuple("conv_config", ["model", 
                                           "in_channels", "out_channels", 
                                           "k", "s", "p", 
                                           "normalize", "activation", "pooling"])


def get_grid(tensor, image_shape):
        
    if len(tensor.shape) == 2:
        tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
    
    return make_grid(tensor, normalize=True)


class VanilaGAN(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_channel,
                 image_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.BCELoss()
        
        
        out_features = image_channel * image_size * image_size
        self.G = MultiLayerPerceptron(hidden_activation=nn.LeakyReLU(0.02),
                                      final_activation=nn.Tanh(),
                                      features=[latent_dim, 256, 512, 1024, out_features])
        self.D = MultiLayerPerceptron(hidden_activation=nn.ReLU(),
                                      final_activation=nn.Sigmoid(),
                                      features=[out_features, 256, 512, 1024, 1])


    def forward(self, z):
        return self.G(z.view(z.size(0), -1))
    
    
    def get_G_loss(self, batch, epoch):
        real, _ = batch
        label_real = torch.ones((real.size(0), 1), device=real.device)
        z = torch.randn((real.size(0), self.latent_dim), device=real.device, requires_grad=True)
        
        fake = self(z)
        
        pred_fake = self.D(fake)
        
        loss_g = self.criterion(pred_fake, label_real)
        return loss_g
        
        
    def get_D_loss(self, batch, epoch):
        real, _ = batch
        label_real = torch.ones((real.size(0), 1), device=real.device)
        label_fake = torch.zeros((real.size(0), 1), device=real.device)
        z = torch.randn((real.size(0), self.latent_dim), device=real.device, requires_grad=True)
        
        fake = self(z)
        
        pred_real = self.D(real.view(real.size(0), -1))
        pred_fake = self.D(fake)
        
        loss_d = ( self.criterion(pred_real, label_real) + self.criterion(pred_fake, label_fake) ) / 2
        return loss_d
    
    
class DCGAN(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_channel,
                 image_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.BCELoss()
        
        
        self.G = BasicConvNet(conv_config=conv_configure(model=DeConvolution_layer,
                                                         in_channels=[latent_dim, 1024, 512, 256, 128],
                                                         out_channels=[1024, 512, 256, 128, image_channel],
                                                         k=[4 for _ in range(5)],
                                                         s=[1,2,2,2,2],
                                                         p=[0,1,1,1,1],
                                                         normalize=[nn.BatchNorm2d(1024), nn.BatchNorm2d(512), nn.BatchNorm2d(256), nn.BatchNorm2d(128), None],
                                                         activation=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Tanh()],
                                                         pooling=[None for _ in range(5)]),
                              image_shape=self.image_shape,
                              output_shape="image")
        
        self.D = BasicConvNet(conv_config=conv_configure(model=Convolution_layer,
                                                         in_channels=[image_channel, 64, 128, 256, 512],
                                                         out_channels=[64, 128, 256, 512, 1],
                                                         k=[4 for _ in range(5)],
                                                         s=[2,2,2,2,1],
                                                         p=[1,1,1,1,0],
                                                         normalize=[None, nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(512), None],
                                                         activation=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.Sigmoid()],
                                                         pooling=[None for _ in range(5)]))


    def forward(self, z):
        return self.G(z)
    
    
    def get_G_loss(self, batch, epoch):
        real, _ = batch
        label_real = torch.ones((real.size(0), 1), device=real.device)
        z = torch.randn((real.size(0), self.latent_dim, 1, 1), device=real.device)
        
        fake = self(z)
        
        pred_fake = self.D(fake).view(real.size(0), -1)
        
        loss_g = self.criterion(pred_fake, label_real)
        return loss_g.requires_grad_(True)
        
        
    def get_D_loss(self, batch, epoch):
        real, _ = batch
        label_real = torch.ones((real.size(0), 1), device=real.device)
        label_fake = torch.zeros((real.size(0), 1), device=real.device)
        z = torch.randn((real.size(0), self.latent_dim, 1, 1), device=real.device)
        
        fake = self(z)
        
        pred_real = self.D(real).view(real.size(0), -1)
        pred_fake = self.D(fake).view(real.size(0), -1)
        
        loss_d = ( self.criterion(pred_real, label_real) + self.criterion(pred_fake, label_fake) ) / 2
        return loss_d.requires_grad_(True)
    
    
class CGAN(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_channel,
                 image_size,
                 label_number):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.BCELoss()
        self.embedding = nn.Embedding(label_number, label_number)
        
        image_feature = image_channel * image_size * image_size
        
        self.G = MultiLayerPerceptron(final_activation=nn.Tanh(),
                                      normalize=nn.BatchNorm1d,
                                      features=[latent_dim + label_number, 128, 256, 512, 1024, image_feature])
        
        self.D = MultiLayerPerceptron(hidden_activation=nn.LeakyReLU(0.02),
                                      final_activation=nn.Sigmoid(),
                                      features=[image_feature + label_number, 512, 512, 512, 1])
    
    
    def forward(self, z, condition):
        z_c = torch.cat((z, condition), -1)
        return self.G(z_c)
    
    
    def get_G_loss(self, batch, epoch):
        real, _ = batch
        label_real = torch.ones((real.size(0), 1), device=real.device)
        z = torch.randn((real.size(0), self.latent_dim), device=real.device, requires_grad=True)
        condition = self.embedding(torch.randint(0, 9, (real.size(0), ), device=real.device))
        
        fake = self(z, condition)
        
        fake_c = torch.cat((fake, condition), -1)
        
        pred_fake = self.D(fake_c)
        
        loss_g = self.criterion(pred_fake, label_real)
        return loss_g.requires_grad_(True)
        
        
    def get_D_loss(self, batch, epoch):
        real, _ = batch
        label_real = torch.ones((real.size(0), 1), device=real.device)
        label_fake = torch.zeros((real.size(0), 1), device=real.device)
        z = torch.randn((real.size(0), self.latent_dim), device=real.device, requires_grad=True)
        condition = self.embedding(torch.randint(0, 9, (real.size(0), ), device=real.device))
        
        fake = self(z, condition)
        
        real_c = torch.cat((real.view(real.size(0), -1), condition), -1)
        fake_c = torch.cat((fake, condition), -1)
        
        pred_real = self.D(real_c)
        pred_fake = self.D(fake_c)
        
        loss_d = ( self.criterion(pred_real, label_real) + self.criterion(pred_fake, label_fake) ) / 2
        return loss_d.requires_grad_(True)
    
    
class WGAN(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_channel,
                 image_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = (image_channel, image_size, image_size)
        
        self.G = BasicConvNet(conv_config=conv_configure(model=DeConvolution_layer,
                                                         in_channels=[latent_dim, 1024, 512, 256, 128],
                                                         out_channels=[1024, 512, 256, 128, image_channel],
                                                         k=[4 for _ in range(5)],
                                                         s=[1,2,2,2,2],
                                                         p=[0,1,1,1,1],
                                                         normalize=[nn.BatchNorm2d(1024), nn.BatchNorm2d(512), nn.BatchNorm2d(256), nn.BatchNorm2d(128), None],
                                                         activation=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Tanh()],
                                                         pooling=[None for _ in range(5)]
                                                         ),
                              image_shape=self.image_shape,
                              output_shape="image")
        
        self.D = BasicConvNet(conv_config=conv_configure(model=Convolution_layer,
                                                         in_channels=[image_channel, 64, 128, 256, 512],
                                                         out_channels=[64, 128, 256, 512, 1],
                                                         k=[4 for _ in range(5)],
                                                         s=[2,2,2,2,1],
                                                         p=[1,1,1,1,0],
                                                         normalize=[None, nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(512), None],
                                                         activation=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.Sigmoid()],
                                                         pooling=[None for _ in range(5)]))


    def forward(self, z):
        return self.G(z)


    def get_G_loss(self, batch, epoch):
        if epoch % 5 == 0:
            real, _ = batch
            z = torch.randn((real.size(0), self.latent_dim, 1, 1), device=real.device)
            
            fake = self.G(z)
            
            pred_fake = self.D(fake).view(real.size(0), -1)
            
            self.loss_g = -torch.mean(pred_fake)
            return self.loss_g.requires_grad_(True)
        else:
            return self.loss_g
        
        
    def get_D_loss(self, batch, epoch):
        real, _ = batch
        z = torch.randn((real.size(0), self.latent_dim, 1, 1), device=real.device)
        
        fake = self.G(z).detach()
        
        pred_real = self.D(real).view(real.size(0), -1)
        pred_fake = self.D(fake).view(real.size(0), -1)
        
        loss_d = -torch.mean(pred_real) + torch.mean(pred_fake)
        return loss_d.requires_grad_(True)
    

class WGAN_GP(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_channel,
                 image_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = (image_channel, image_size, image_size)
        
        # Loss weight for gradient penalty
        self.lambda_gp = 10

        self.G = BasicConvNet(conv_config=conv_configure(model=DeConvolution_layer,
                                                         in_channels=[latent_dim, 1024, 512, 256, 128],
                                                         out_channels=[1024, 512, 256, 128, image_channel],
                                                         k=[4 for _ in range(5)],
                                                         s=[1,2,2,2,2],
                                                         p=[0,1,1,1,1],
                                                         normalize=[nn.BatchNorm2d(1024), nn.BatchNorm2d(512), nn.BatchNorm2d(256), nn.BatchNorm2d(128), None],
                                                         activation=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Tanh()],
                                                         pooling=[None for _ in range(5)]
                                                         ),
                              image_shape=self.image_shape,
                              output_shape="image")
        
        self.D = BasicConvNet(conv_config=conv_configure(model=Convolution_layer,
                                                         in_channels=[image_channel, 64, 128, 256, 512],
                                                         out_channels=[64, 128, 256, 512, 1],
                                                         k=[4 for _ in range(5)],
                                                         s=[2,2,2,2,1],
                                                         p=[1,1,1,1,0],
                                                         normalize=[None, nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(512), None],
                                                         activation=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.Sigmoid()],
                                                         pooling=[None for _ in range(5)]
                                                         ),
                              image_shape=self.image_shape,
                              output_shape="scalar")
    
    def compute_gp(self, real_sample, fake_sample):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn((real_sample.size(0), 1, 1, 1), device=real_sample.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_sample + ((1 - alpha) * fake_sample))
        d_interpolates = self.D(interpolates).requires_grad_(True)
        fake = torch.ones((real_sample.size(0), 1, 1, 1), device=real_sample.device)
        # Get gradient w.r.t. interpolates
        gradients, *_ = torch.autograd.grad(d_interpolates, interpolates, fake, create_graph=True)
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def get_G_loss(self, batch, epoch):
        if epoch % 5 == 0:
            real, _ = batch
            z = torch.randn((real.size(0), self.latent_dim, 1, 1), device=real.device)
            
            fake = self.G(z)
            
            pred_fake = self.D(fake).view(real.size(0), -1)
            
            self.loss_g = -torch.mean(pred_fake)
            return self.loss_g.requires_grad_(True)
        else:
            return self.loss_g
        
        
    def get_D_loss(self, batch, epoch):
        with torch.enable_grad():
            real, _ = batch
            z = torch.randn((real.size(0), self.latent_dim, 1, 1), requires_grad=True, device=real.device)
            
            fake = self.G(z)
            
            pred_real = self.D(real).view(real.size(0), -1)
            pred_fake = self.D(fake).view(real.size(0), -1)
            
            gradient_penalty = self.compute_gp(real, fake)
        
        loss_d = -torch.mean(pred_real) + torch.mean(pred_fake) + self.lambda_gp * gradient_penalty
        return loss_d.requires_grad_(True)


    def forward(self, z):
        return self.G(z)
    

class WGAN_DIV(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 image_channel,
                 image_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = (image_channel, image_size, image_size)

        self.k, self.p = 2, 6
        
        self.G = BasicConvNet(conv_config=conv_configure(model=DeConvolution_layer,
                                                         in_channels=[latent_dim, 1024, 512, 256, 128],
                                                         out_channels=[1024, 512, 256, 128, image_channel],
                                                         k=[4 for _ in range(5)],
                                                         s=[1,2,2,2,2],
                                                         p=[0,1,1,1,1],
                                                         normalize=[nn.BatchNorm2d(1024), nn.BatchNorm2d(512), nn.BatchNorm2d(256), nn.BatchNorm2d(128), None],
                                                         activation=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Tanh()],
                                                         pooling=[None for _ in range(5)]
                                                         ),
                              image_shape=self.image_shape,
                              output_shape="image")
        
        self.D = BasicConvNet(conv_config=conv_configure(model=Convolution_layer,
                                                         in_channels=[image_channel, 64, 128, 256, 512],
                                                         out_channels=[64, 128, 256, 512, 1],
                                                         k=[4 for _ in range(5)],
                                                         s=[2,2,2,2,1],
                                                         p=[1,1,1,1,0],
                                                         normalize=[None, nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(512), None],
                                                         activation=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.Sigmoid()],
                                                         pooling=[None for _ in range(5)]
                                                         ))
    
    
    def compute_div_gp(self, real, fake):
        real_out = torch.ones((real.size(0), 1), device=real.device)
        
        real_grad = torch.autograd(self.D(real), real, real_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (self.p / 2)
        
        fake_out = torch.ones((real.size(0), 1), device=real.device)
        
        fake_grad = torch.autograd(self.D(fake), fake, fake_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (self.p / 2)
        
        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * self.k / 2
        return div_gp


    def forward(self, z):
        return self.G(z)
    
    
    def get_G_loss(self, batch, epoch):
        if epoch % 5 == 0:
            real, _ = batch
            z = torch.randn((real.size(0), self.latent_dim, 1, 1), device=real.device)
            
            fake = self.G(z)
            
            pred_fake = self.D(fake).view(real.size(0), -1)
            
            self.loss_g = -torch.mean(pred_fake)
            return self.loss_g.requires_grad_(True)
        else:
            return self.loss_g
        
        
    def get_D_loss(self, batch, epoch):
        with torch.enable_grad():
            real, _ = batch
            z = torch.randn((real.size(0), self.latent_dim, 1, 1), device=real.device, requires_grad=True)
            
            fake = self.G(z)
            
            pred_real = self.D(real).view(real.size(0), -1)
            pred_fake = self.D(fake).view(real.size(0), -1)
            
            div_gp = self.compute_div_gradient_penalty(pred_real, pred_fake)
        
        loss_d = -torch.mean(pred_real) + torch.mean(pred_fake) + div_gp
        return loss_d.requires_grad_(True)


class Pix2Pix(nn.Module):
    pass
    
    
class Lit_gan(pl.LightningModule):
    def __init__(self,
                 lr,
                 optim_name: str,
                 model_name,
                 model_args):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
        
    def configure_optimizers(self):
        optim_g = self.optimizer(self.model.G.parameters(), self.lr)
        optim_d = self.optimizer(self.model.D.parameters(), self.lr)
        
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g, step_size=1)
        lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, step_size=1)
        
        return [optim_g, optim_d], [lr_scheduler_g, lr_scheduler_d]
       
       
    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()
        
        # Training Generator
        self.toggle_optimizer(optim_g)
        
        loss_g = self.model.get_G_loss(batch, self.current_epoch)
        self.log("loss_g", loss_g, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_g)
        optim_g.step()
        optim_g.zero_grad()
        
        self.untoggle_optimizer(optim_g)
        
        # Training Discriminator
        self.toggle_optimizer(optim_d)
        
        loss_d = self.model.get_D_loss(batch, self.current_epoch)
        self.log("loss_d", loss_d, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_d)
        optim_d.step()
        optim_d.zero_grad()
        
        self.untoggle_optimizer(optim_d)
        
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        # return super().on_train_batch_start(batch, batch_idx)
        pass
    
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        z = torch.randn((batch[0].size(0), self.model.latent_dim, 1, 1), device=batch[0].device)
        self.logger.experiment.add_image("fake", get_grid(self.model(z), self.model.image_shape), self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()
        
        # Training Generator
        self.toggle_optimizer(optim_g)
        
        loss_g = self.model.get_G_loss(batch, self.current_epoch)
        self.log("loss_g", loss_g, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_g)
        optim_g.step()
        optim_g.zero_grad()
        
        self.untoggle_optimizer(optim_g)
        
        # Training Discriminator
        self.toggle_optimizer(optim_d)
        
        loss_d = self.model.get_D_loss(batch, self.current_epoch)
        self.log("loss_d", loss_d, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_d)
        optim_d.step()
        optim_d.zero_grad()
        
        self.untoggle_optimizer(optim_d)
    
    
    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        pass
    
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
        pass
    
    
    def test_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()
        
        # Training Generator
        self.toggle_optimizer(optim_g)
        
        loss_g = self.model.get_G_loss(batch, self.current_epoch)
        self.log("loss_g", loss_g, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_g)
        optim_g.step()
        optim_g.zero_grad()
        
        self.untoggle_optimizer(optim_g)
        
        # Training Discriminator
        self.toggle_optimizer(optim_d)
        
        loss_d = self.model.get_D_loss(batch, self.current_epoch)
        self.log("loss_d", loss_d, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_d)
        optim_d.step()
        optim_d.zero_grad()
        
        self.untoggle_optimizer(optim_d)
    
    
    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_test_batch_start(batch, batch_idx, dataloader_idx)
        pass
    
    
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)
        pass
    
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().predict_step(batch, batch_idx, dataloader_idx)
        pass