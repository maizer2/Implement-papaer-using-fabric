import importlib, os
from collections import namedtuple
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim as toptim
import torch.nn.functional as F

from models.cnn.CNN import BasicConvNet, DeConvolution_layer, Convolution_layer
from models.mlp.MLP import MultiLayerPerceptron

conv_configure= namedtuple("conv_config", ["model", 
                                           "in_channels", "out_channels", 
                                           "k", "s", "p", 
                                           "normalize", "activation", "pooling"])


def visualization(tensor, image_shape = None, out_path = None):
    if out_path is None:
        out_path = "./"
        
    from torchvision.utils import make_grid
    from torchvision import transforms
    
    if len(tensor.shape) == 2:
        tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
    
    image_grid = make_grid(tensor, normalize=True)
    print(image_grid.shape)
    transforms.ToPILImage()(image_grid).save(os.path.join(out_path, "test.png"))
    
    
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


    def forward(self):
        return self.G(torch.randn(self.batch_size, self.latent_dim).cuda())
    
    
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
                                                         pooling=[None for _ in range(5)]
                                                         ),
                              img_size=image_size,
                              output_shape="image",
                              out_channels=1)
        
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
                              img_size=image_size,
                              output_shape="scalar",
                              out_channels=1)
    
    def get_G_loss(self, batch_size):
        self.batch_size = batch_size
        z = torch.randn(batch_size, self.latent_dim, 1, 1).cuda()
        fake = self.G(z)
        
        label_real = torch.ones(batch_size, 1).cuda()
        pred_fake = self.D(fake).view(batch_size, -1)
        loss_g = self.criterion(pred_fake, label_real)
        return loss_g.requires_grad_(True)
        
        
    def get_D_loss(self, batch_size, real):
        z = torch.randn(batch_size, self.latent_dim, 1, 1).cuda()
        fake = self.G(z).detach()
        
        label_real = torch.ones(batch_size, 1).cuda()
        label_fake = torch.zeros(batch_size, 1).cuda()
        pred_real = self.D(real).view(batch_size, -1)
        pred_fake = self.D(fake).view(batch_size, -1)
        
        loss_d = ( self.criterion(pred_real, label_real).requires_grad_(True) + self.criterion(pred_fake, label_fake).requires_grad_(True) ) / 2
        return loss_d.requires_grad_(True)


    def forward(self):
        return self.G(torch.randn(self.batch_size, self.latent_dim, 1, 1).cuda())
    
    
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
    
    
    def get_G_loss(self, batch_size):
        self.batch_size = batch_size
        z = torch.randn(batch_size, self.latent_dim).cuda()
        condition = self.embedding(torch.randint(0, 9, (batch_size, )).cuda())
        z_c = torch.cat((z, condition), -1)
        fake = self.G(z_c)
        
        fake_c = torch.cat((fake, condition), -1)
        
        label_real = torch.ones(batch_size, 1).cuda()
        pred_fake = self.D(fake_c)
        
        loss_g = self.criterion(pred_fake, label_real)
        return loss_g.requires_grad_(True)
        
        
    def get_D_loss(self, batch_size, real):
        z = torch.randn(batch_size, self.latent_dim).cuda()
        condition = self.embedding(torch.randint(0, 9, (batch_size, )).cuda())
        z_c = torch.cat((z, condition), -1)
        fake = self.G(z_c).detach()
        
        real_c = torch.cat((real.view(batch_size, -1), condition), -1)
        fake_c = torch.cat((fake, condition), -1)
        
        label_real = torch.ones(batch_size, 1).cuda()
        label_fake = torch.zeros(batch_size, 1).cuda()
        pred_real = self.D(real_c)
        pred_fake = self.D(fake_c)
        
        loss_d = ( self.criterion(pred_real, label_real).requires_grad_(True) + self.criterion(pred_fake, label_fake).requires_grad_(True) ) / 2
        return loss_d.requires_grad_(True)
    
    
    def forward(self):
        return self.G(torch.randn(self.batch_size, self.latent_dim).cuda())
    
    
class LitGAN(pl.LightningModule):
    def __init__(self,
                 lr, 
                 model_name,
                 model_args):
        super().__init__()
        
        self.automatic_optimization = False
        self.lr = lr
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
        
    def get_loss(self, batch, log_g_string, log_d_string):
        optim_g, optim_d = self.optimizers()
        real, _ = batch
        batch_size = real.size(0)
        
        # Training Generator
        self.toggle_optimizer(optim_g)
        
        loss_g = self.model.get_G_loss(batch_size)
        self.log(log_g_string, loss_g, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_g)
        optim_g.step()
        optim_g.zero_grad()
        
        self.untoggle_optimizer(optim_g)
        
        # Training Discriminator
        self.toggle_optimizer(optim_d)
        
        loss_d = self.model.get_D_loss(batch_size, real)
        self.log(log_d_string, loss_d, prog_bar=True, sync_dist=True)
        self.manual_backward(loss_d)
        optim_d.step()
        optim_d.zero_grad()
        
        self.untoggle_optimizer(optim_d)
        
        
        
    def configure_optimizers(self):
        optim_g = toptim.Adam(self.model.G.parameters(), self.lr)
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g, step_size=1)
        optim_d = toptim.Adam(self.model.D.parameters(), self.lr)
        lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, step_size=1)
        return [optim_g, optim_d], [lr_scheduler_g, lr_scheduler_d]
       
       
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, "loss_g", "loss_d")
    
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        # return super().on_train_batch_start(batch, batch_idx)
        pass
    
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        visualization(self.model(), self.model.image_shape, )
        pass
    
    
    def validation_step(self, batch, batch_idx):
        return self.get_loss(batch, "loss_g", "loss_d")
    
    
    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        pass
    
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
        pass
    
    
    def test_step(self, batch, batch_idx):
        return self.get_loss(batch, "loss_g", "loss_d")
    
    
    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_test_batch_start(batch, batch_idx, dataloader_idx)
        pass
    
    
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)
        pass
    
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # return super().predict_step(batch, batch_idx, dataloader_idx)
        pass