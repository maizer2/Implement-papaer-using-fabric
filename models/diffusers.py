import importlib, os
from typing import Any, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms


def list_to_tuple(data):
    if isinstance(data, list):
        return tuple(data)
    elif isinstance(data, int):
        return (data, data)
    return data


class DDPM(nn.Module):

    def __init__(self,
                 scheduler_name: str = "DDPMScheduler",
                 sample_size: Optional[Union[tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 1_000,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import DDPM library
        from models.diffusion.pipeline.custom_DDPMPipeline import DDPMPipeline
        from diffusers.models import UNet2DModel
        
        self.criterion = nn.MSELoss()
        
        self.num_sampling = num_sampling
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        self.unet = UNet2DModel(sample_size=list_to_tuple(sample_size),
                                in_channels=in_channels,
                                out_channels=out_channels,
                                down_block_types=("DownBlock2D",
                                                "DownBlock2D",
                                                "DownBlock2D",
                                                "DownBlock2D",
                                                "AttnDownBlock2D",
                                                "DownBlock2D"),
                                up_block_types=("UpBlock2D",
                                                "AttnUpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D"),
                                block_out_channels=(128, 128, 256, 256, 512, 512),
                                layers_per_block=2)
        
        self.scheduler = getattr(importlib.import_module("diffusers.schedulers"), scheduler_name)(num_train_steps)
        self.pipeline = DDPMPipeline(self.unet, self.scheduler)
       
    
    # Sampling
    def reverse_latent_diffusion_process(self, xT = None) -> torch.FloatTensor:
        pred_x0 = self.pipeline(xT=xT,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_x0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        if x0 is not None:
            xT = self.forward_latent_diffusion_process(x0)
        else:
            xT = None
        
        pred_x0 = self.reverse_latent_diffusion_process(xT)
        
        return pred_x0
    
    
    def get_loss(self, x0):
        t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        xT = self.forward_latent_diffusion_process(x0, noise, t)
        eps_theta = self.unet(xT, t).sample
        
        loss = self.criterion(noise, eps_theta)
        return loss


class UnconditionalLDM(nn.Module):

    def __init__(self,
                 sample_size: Optional[Union[list, tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 50,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import LDM library
        from models.diffusion.pipeline.custom_LDMPipeline import UnconditionalLDMPipeline
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.models.vq_model import VQModel
        from diffusers.models.unet_2d import UNet2DModel
        
        self.criterion = nn.MSELoss()
        
        self.sample_size = list_to_tuple(sample_size)
        self.num_sampling = num_sampling
        
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        # self.vqvae = VQModel(in_channels=in_channels,
        #                      out_channels=out_channels,
        #                      down_block_types=("DownEncoderBlock2D",    #256 -> 128 -> 64 -> 32 / 192 -> 96 -> 48 -> 24
        #                                        "DownEncoderBlock2D",
        #                                        "DownEncoderBlock2D",
        #                                        "DownEncoderBlock2D"),
        #                      up_block_types=("UpDecoderBlock2D",
        #                                      "UpDecoderBlock2D",
        #                                      "UpDecoderBlock2D",
        #                                      "UpDecoderBlock2D",),
        #                      block_out_channels=(64, 128, 256, 512),
        #                      layers_per_block=1,
        #                      latent_channels=out_channels,
        #                      sample_size=sample_size[0])
        self.vqvae = VQModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", ignore_mismatched_sizes=True)
        self.unet = UNet2DModel(sample_size=(32, 24),
                                in_channels=out_channels,
                                out_channels=out_channels,
                                down_block_types=("DownBlock2D",    # 32 -> 16 -> 8 -> 4 / 24 -> 12 -> 6 -> 3
                                                "AttnDownBlock2D",
                                                "DownBlock2D"),
                                up_block_types=("UpBlock2D",
                                                "AttnUpBlock2D",
                                                "UpBlock2D"),
                                block_out_channels=(128, 256, 512),
                                layers_per_block=2)
        
        self.scheduler = DDIMScheduler(self.num_train_steps)
        self.pipeline = UnconditionalLDMPipeline(vqvae=self.vqvae,
                                                 unet=self.unet,
                                                 scheduler=self.scheduler)
        
    
    def encode(self, x0):
        return self.vqvae.encode(x0).latents
    
    
    def decode(self, z0):
        return self.vqvae.decode(z0).sample
    
    
    # Sampling
    def reverse_latent_diffusion_process(self, latents = None) -> torch.FloatTensor:
        pred_z0 = self.pipeline(latents=latents,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_z0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        if x0 is not None:
            z = self.encode(x0)
            zT = self.forward_latent_diffusion_process(z)
        else:
            zT = None
        
        pred_z0 = self.reverse_latent_diffusion_process(zT)
        pred_x0 = self.decode(pred_z0)
        
        return pred_x0
    
    
    def get_loss(self, x0):
        z = self.encode(x0)
        t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        noise = torch.randn(z.shape, dtype=x0.dtype, device=x0.device)
        
        zT = self.forward_latent_diffusion_process(z, noise, t)
        print(zT.shape)
        eps_theta = self.unet(zT, t).sample
        print(eps_theta.shape)
        exit()
        loss = self.criterion(noise, eps_theta)
        return loss


class LDM(nn.Module):

    def __init__(self,
                 sample_size: Optional[Union[list, tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 50,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import LDM library
        from diffusers import StableDiffusionImg2ImgPipeline
        
        self.criterion = nn.MSELoss()
        self.scheduler = None
        self.vae = None
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        
    
    def encode(self, x0):
        return self.vae.encode(x0).latents
    
    
    def decode(self, z0):
        return self.vae.decode(z0).sample
    
    
    # Sampling
    def reverse_latent_diffusion_process(self, latents = None) -> torch.FloatTensor:
        pred_z0 = self.pipeline(latents=latents,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_z0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        pass
    
    
    def get_loss(self, x0):
        pipe = self.pipeline.cuda()
        
        from torchvision import transforms
        
        x0 = transforms.Resize((256, 256))(x0)
        print(x0.shape)
        exit()
 

class DDIM_on_StableDiffusion(nn.Module):

    def __init__(self,
                 sample_size: Optional[Union[list, tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 50,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import LDM library
        from models.diffusion.pipeline.custom_StableDiffusionPipeline import StableDiffusionPipeline
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.models.vq_model import VQModel
        from diffusers.models.unet_2d import UNet2DModel
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        
        from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
        
        self.criterion = nn.MSELoss()
        
        self.sample_size = list_to_tuple(sample_size)
        self.num_sampling = num_sampling
        
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.vae = VQModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", ignore_mismatched_sizes=True)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.unet = UNet2DModel(sample_size=(32, 24),
                                in_channels=out_channels,
                                out_channels=out_channels,
                                down_block_types=("DownBlock2D",    # 32 -> 16 -> 8 -> 4 / 24 -> 12 -> 6 -> 3
                                                "AttnDownBlock2D",
                                                "DownBlock2D"),
                                up_block_types=("UpBlock2D",
                                                "AttnUpBlock2D",
                                                "UpBlock2D"),
                                block_out_channels=(128, 256, 512),
                                layers_per_block=2)
        self.scheduler = DDIMScheduler(self.num_train_steps)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4")
        self.feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.pipeline = StableDiffusionPipeline(vae=self.vae,
                                                text_encoder=self.text_encoder,
                                                tokenizer=self.tokenizer,
                                                unet=self.unet,
                                                scheduler=self.scheduler,
                                                safety_checker=self.safety_checker,
                                                feature_extractor=self.feature_extractor,
                                                requires_safety_checker=False
                                                )
        
    
    def encode(self, x0):
        return self.vqvae.encode(x0).latents
    
    
    def decode(self, z0):
        return self.vqvae.decode(z0).sample
    
    
    # Sampling
    def reverse_latent_diffusion_process(self, latents = None) -> torch.FloatTensor:
        pred_z0 = self.pipeline(latents=latents,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_z0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        if x0 is not None:
            z = self.encode(x0)
            zT = self.forward_latent_diffusion_process(z)
        else:
            zT = None
        
        pred_z0 = self.reverse_latent_diffusion_process(zT)
        pred_x0 = self.decode(pred_z0)
        
        return pred_x0
    
    
    def get_loss(self, x0):
        z = self.encode(x0)
        t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        noise = torch.randn(z.shape, dtype=x0.dtype, device=x0.device)
        
        zT = self.forward_latent_diffusion_process(z, noise, t)
        print(zT.shape)
        eps_theta = self.unet(zT, t).sample
        print(eps_theta.shape)
        exit()
        loss = self.criterion(noise, eps_theta)
        return loss
    
    
class Lit_diffusers(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 img2img: bool,
                 sampling_step: int,
                 optim_name: str,
                 model_name: str,
                 model_args: tuple) -> None:
        super().__init__()
        self.lr = lr
        self.img2img = img2img
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
    
        
        
    def get_input(self, batch) -> torch.Tensor:
        image, cm = batch['image'], batch['cloth_mask']
        x0 = torch.cat([image, cm], 1)
        
        return x0
        
    
    def get_condition(self, batch) -> list:
        condition = [batch["shape"], batch["head"], batch["pose_image"], batch["cloth"]]
        
        return condition
        
    
    def get_tryon(self, cloth, p_rendered, m_composite):
        p_tryon = cloth * m_composite + p_rendered * (1 - m_composite)
        
        return p_tryon
    
    
    def sampling(self, batch):
        # with torch.no_grad():
        #     if self.img2img:
        #         x0 = x0[:self.model.num_sampling]
        #         self.logger.experiment.add_image("x0", self.get_grid(x0), self.current_epoch)
                
        #     else:
        #         x0 = None
                
        #     pred_x0 = self.model(x0)
            
        #     self.logger.experiment.add_image("pred_x0", self.get_grid(pred_x0), self.current_epoch)
        
        with torch.no_grad():
            outputs = self.model(self.get_input(batch))

            p_rendered, m_composite = torch.split(outputs, 3, 1)
            tryon = self.get_tryon(batch["cloth"], p_rendered, m_composite)
                
            self.logger.experiment.add_image("Original", self.get_grid(batch["image"]), self.current_epoch)
            self.logger.experiment.add_image("Rendered", self.get_grid(p_rendered), self.current_epoch)
            self.logger.experiment.add_image("Cloth_mask", self.get_grid(batch["cloth_mask"]), self.current_epoch)
            self.logger.experiment.add_image("Composite", self.get_grid(m_composite), self.current_epoch)
            self.logger.experiment.add_image("Tryon", self.get_grid(tryon), self.current_epoch)
    
    
    def training_step(self, batch, batch_idx):
        x0 = self.get_input(batch)
        
        loss = self.model.get_loss(x0)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        
        if self.trainer.is_last_batch:
            if self.current_epoch == 0:
                self.sampling(batch)
            elif (self.current_epoch + 1) % self.sampling_step == 0:
                self.sampling(batch)
                    
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x0, _ = batch
        loss = self.model.get_loss(x0)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x0, _ = batch
        loss = self.model.get_loss(x0)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x0, _ = batch
        self.sampling(x0)
    