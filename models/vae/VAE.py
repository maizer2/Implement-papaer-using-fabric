import importlib
from typing import Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn

from torchvision.utils import make_grid

# class AutoencoderKL(nn.Module):
#     def __init__(self,
#                  num_sampling = 3,
#                  in_channels = 4,
#                  out_channels = 4,
#                  sample_size:Optional[Union[list, tuple, int]] = (256, 192)):
#         super().__init__()
#         self.criterion = nn.Identity()
        
#         from models.vae.autoencoder_kl import AutoencoderKL
        
#         self.num_sampling = num_sampling
#         self.vae = AutoencoderKL(in_channels=in_channels,
#                                     out_channels=out_channels,
#                                     down_block_types=("DownEncoderBlock2D",    #256 -> 128 -> 64 -> 32 / 192 -> 96 -> 48 -> 24
#                                                     "DownEncoderBlock2D",
#                                                     "DownEncoderBlock2D",
#                                                     "DownEncoderBlock2D",
#                                                     "DownEncoderBlock2D"),
#                                     up_block_types=("UpDecoderBlock2D",
#                                                     "UpDecoderBlock2D",
#                                                     "UpDecoderBlock2D",
#                                                     "UpDecoderBlock2D",
#                                                     "UpDecoderBlock2D",),
#                                     block_out_channels=(64, 128, 256, 512, 512),
#                                     layers_per_block=1,
#                                     latent_channels=out_channels,
#                                     sample_size=sample_size[0])
    
#     def forward(self, x0):
#         return self.vae.forward(x0).sample
    
    
#     def get_loss(self, x0, epoch, device):
#         loss = self.vae.get_loss(x0)
#         return loss
        

class VQ_VAE(nn.Module):
    def __init__(self,
                 num_sampling = 3,
                 in_channels = 4,
                 out_channels = 4,
                 sample_size:Optional[Union[list, tuple, int]] = (256, 192)):
        super().__init__()
        self.L2criterion = nn.MSELoss()
        self.L1criterion = nn.L1Loss()
        # self.BCEcriterion = nn.BCELoss()
        
        from models.vae.vq_model import VQModel
        
        self.num_sampling = num_sampling
        self.vae = VQModel(in_channels=in_channels,
                             out_channels=out_channels,
                             down_block_types=("DownEncoderBlock2D",    #256 -> 128 -> 64 -> 32 / 192 -> 96 -> 48 -> 24
                                               "DownEncoderBlock2D",
                                               "DownEncoderBlock2D",
                                               "DownEncoderBlock2D",
                                               "DownEncoderBlock2D"),
                             up_block_types=("UpDecoderBlock2D",
                                             "UpDecoderBlock2D",
                                             "UpDecoderBlock2D",
                                             "UpDecoderBlock2D",
                                             "UpDecoderBlock2D",),
                             block_out_channels=(64, 128, 256, 512, 512),
                             layers_per_block=1,
                             latent_channels=out_channels,
                             sample_size=sample_size[0])
    
    def forward(self, x0):
        return self.vae.forward(x0).sample
    
    
    def get_loss(self, x0, epoch, device):
        loss = self.vae.get_loss(x0)
        return loss
        
    
class LitVAE(pl.LightningModule):
    def __init__(self,
                 lr,
                 sampling_step,
                 optim_name,
                 model_name,
                 model_args):
        super().__init__()
        self.lr = lr
        self.sampling_step = sampling_step
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
    
    
    def get_grid(self, tensor, image_shape = None):
        
        if len(tensor.shape) == 2:
            tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
        
        return make_grid(tensor, normalize=True)
    
        
    def configure_optimizers(self):
        optim = self.optimizer(self.model.parameters(), self.lr)
        
        return {"optimizer": optim}
       
    
    def sampling(self, x0):
        with torch.no_grad():
            x0 = x0[:self.model.num_sampling]
            p_rendered, m_composite = torch.split(x0, 3,1)
            self.logger.experiment.add_image("cloth_mask", self.get_grid(m_composite), self.current_epoch)
            self.logger.experiment.add_image("image", self.get_grid(p_rendered), self.current_epoch)
            
            pred_x0 = self.model(x0)
            
            p_rendered, m_composite = torch.split(pred_x0, 3,1)
            self.logger.experiment.add_image("m_composite", self.get_grid(m_composite), self.current_epoch)
            self.logger.experiment.add_image("p_rendered", self.get_grid(p_rendered), self.current_epoch)
            
            
    def training_step(self, batch, batch_idx):
        image = batch['image']
        mask = batch["cloth_mask"]
        x0 = torch.cat([image, mask], 1)
        
        loss = self.model.get_loss(x0, self.current_epoch, self.device)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        
        if self.trainer.is_last_batch:
            if self.current_epoch == 0:
                self.sampling(x0)
            elif (self.current_epoch + 1) % self.sampling_step == 0:
                self.sampling(x0)
                
        return loss
    
    
    # def validation_step(self, batch, batch_idx):
    #     loss = self.model.get_loss(batch, self.current_epoch, self.device)
    #     self.log("val_loss", loss, prog_bar=True, sync_dist=True)
    #     return loss
    
    
    # def test_step(self, batch, batch_idx):
    #     loss = self.model.get_loss(batch, self.current_epoch, self.device)
    #     self.log("test_loss", loss, prog_bar=True, sync_dist=True)
    #     return loss
    