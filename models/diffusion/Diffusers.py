import importlib, os
from collections import namedtuple
from typing import Any, Optional
from tqdm import tqdm

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

from diffusers import DDPMPipeline, DDPMScheduler, DDIMPipeline, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


class DDPM(nn.Module):
    def __init__(self,
                 eps_model_name:str,
                 eps_model_args:dict,
                 image_channel=3,
                 image_size=32,
                 n_steps=1_000):
        super().__init__()
        self.image_shape = (image_channel, image_size, image_size)
        self.criterion = nn.MSELoss()
        self.n_steps = n_steps
        
        eps_model_args.update({"sample_size": image_size, 
                               "in_channels": image_channel,
                               "out_channels": image_channel,
                               "block_out_channels": (128, 128, 256, 256, 512, 512),
                               "down_block_types": (
                                    "DownBlock2D",  # a regular ResNet downsampling block
                                    "DownBlock2D", 
                                    "DownBlock2D", 
                                    "DownBlock2D", 
                                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                                    "DownBlock2D"),
                                "up_block_types":(
                                    "UpBlock2D",  # a regular ResNet upsampling block
                                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                                    "UpBlock2D", 
                                    "UpBlock2D", 
                                    "UpBlock2D", 
                                    "UpBlock2D")
                               })
        
        self.eps_model = get_obj_from_str(eps_model_name)(**eps_model_args)
        self.scheduler = DDPMScheduler(n_steps)
        self.pipeline = DDPMPipeline(unet=self.eps_model,
                                     scheduler=self.scheduler)
    
    
    def forward(self, batch):
        x0, _ = batch
        noise = torch.randn(x0.shape)
        timesteps = torch.FloatTensor([50])
        xT = self.scheduler.add_noise(x0, noise, timesteps)
        
        x0_hat = xT
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noisy_x = self.eps_model(x0_hat, t).sample
            previous_noisy_sample = self.scheduler.step(noisy_x, t, input).prev_sample
            x0_hat = previous_noisy_sample
        
        return x0_hat
    
    
    def get_loss(self, batch, epoch):
        x0, _ = batch
        noise = torch.randn(x0.shape)
        timesteps = torch.LongTensor([self.n_steps])
        
        xT = self.scheduler.add_noise(x0, noise, timesteps)
        print(xT.shape, noise.shape)
        exit()
        loss = self.criterion(xT, noise)
        
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
    

class LitDiffusers(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 lr_warmup_steps: int,
                 optim_name: str,
                 model_name: str,
                 model_args: tuple) -> None:
        super().__init__()
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
        
    def get_grid(self, tensor, image_shape):
        
        if len(tensor.shape) == 2:
            tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
        
        return make_grid(tensor, normalize=True)
    
    
    def configure_optimizers(self):
        optim = self.optimizer(self.model.eps_model.parameters(), self.lr)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=(625 * self.trainer.max_epochs)
        )
        return {"optimizer":optim, 
                "lr_scheduler":scheduler}
    
        # return optim
    
    def training_step(self, batch, batch_idx):
        # self.model.set_variable_device(self.device)
        loss = self.model.get_loss(batch, self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        self.logger.experiment.add_image("x0_hat", self.get_grid(self.model(batch), self.model.image_shape), self.current_epoch)
    
    
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
        x0_hat = self.model(batch)
        return x0_hat
    
    
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        out_path = os.path.join(self.trainer.log_dir, "output_predict")
        os.makedirs(out_path, exist_ok=True)
        
        x_hat_grid = self.get_grid(outputs, outputs.shape)
        x_hat_PIL = transforms.ToPILImage()(x_hat_grid)
        x_hat_PIL.save(os.path.join(out_path, f"{batch_idx}.png"))