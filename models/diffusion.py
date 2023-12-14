import importlib, os
from typing import Any, Optional, Union, Dict, 

import lightning.pytorch as pl

import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import make_grid
from torchvision import transforms

from run import get_obj_from_str, instantiate_from_config

from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, StableDiffusionControlNetPipeline, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor

from models.base import Lit_base, Module_base

class DDPM(nn.Module):
    def __init__(self,
                 optim_name: str,
                 unet_config: str,
                 num_train_timesteps = 1_000,
                 num_inference_steps = 1_000,
                 model_path: str = None):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = nn.MSELoss()
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.model_path = model_path
        
        self.unet = instantiate_from_config(unet_config)
        
        self.beta = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta
    
        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
            
    def class_instance_to_cuda(self):
        self.beta = self.beta.cuda()
        self.alpha = self.alpha.cuda()
        self.alpha_bar = self.alpha_bar.cuda()
        self.sigma2 = self.sigma2.cuda()
        
    def gather(self, consts: torch.Tensor, t: torch.Tensor):
        return consts.gather(-1, t).reshape(-1, 1, 1, 1)
        
    def q_xt_x0(self, x0, t):
        mean = self.gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - self.gather(self.alpha_bar, t)
        
        return mean, var
        
    def forward_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        if t is None:
            t = torch.full((x0.size(0), ), (self.num_inference_steps - 1), dtype=torch.long, device=x0.device)
            
        mean, var = self.q_xt_x0(x0, t)

        xT = mean + (var ** 0.5) * noise
        
        return xT
    
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.unet(xt, t).sample
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gather(self.sigma2, t)
        eps = torch.randn(xt.shape).cuda()
        
        return mean + (var ** .5) * eps
        
    def reverse_diffusion_process(self, xT = None, shape = None) -> torch.FloatTensor:
        if xT is None:
            xT = torch.randn(shape, dtype=torch.float32).cuda()
        
        pred_x0 = xT
        for t_ in range(self.num_inference_steps):
            t = torch.tensor([self.num_inference_steps - (t_ + 1)], dtype=torch.long).cuda()
            
            # 1. predict noise model_output
            # 2. compute previous image: x_t -> x_t-1
            pred_x0 = self.p_sample(pred_x0, t)
        
        return pred_x0
    
    def forward(self, x0, noise):
        t = torch.randint(0, self.num_train_timesteps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        xT = self.forward_diffusion_process(x0, noise, t)
        
        eps_theta = self.unet(xT, t).sample
        
        return eps_theta
    
    def get_input(self, batch, num_sampling = None):
        x0, text = batch
        
        if num_sampling is not None:
            x0 = x0[:num_sampling]
            text = text[:num_sampling]
        
        return x0, text
    
    def get_loss(self, batch):
        x0, text = self.get_input(batch)
        self.class_instance_to_cuda()
        
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        rec_sample = self(x0, noise)
        
        loss = self.criterion(noise, rec_sample)
        
        return loss
    
    def inference(self, batch, num_sampling, img2img = True):
        self.class_instance_to_cuda()
        self.unet.eval()
        
        with torch.no_grad():
            x0, _ = self.get_input(batch, num_sampling)
            
            if img2img:
                xT = self.forward_diffusion_process(x0)
            else:
                xT = None
            
            pred_x0 = self.reverse_diffusion_process(xT, x0.shape)
        
        self.unet.train()
        
        return pred_x0
    
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
class diffusers_DDPM(nn.Module):
    def __init__(self,
                 optim_name: str,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 1_000,
                 model_path: str = None):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        self.model_path = model_path
        
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
            
    def forward_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        if t is None:
            t = torch.full((x0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=x0.device)
            
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    def reverse_diffusion_process(self, xT = None, shape = None) -> torch.FloatTensor:
        if xT is None:
            xT = torch.randn(shape, dtype=torch.float32).cuda()
        
        pred_x0 = xT
        self.scheduler.set_timesteps(self.num_inference_steps, xT.device)
        for t in self.scheduler.timesteps:
            
            # 1. predict noise model_output
            model_output = self.unet(pred_x0, t).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_x0 = self.scheduler.step(model_output, t, pred_x0).prev_sample
        
        return pred_x0
    
    def forward(self, x0, noise):
        t = torch.randint(0, len(self.scheduler.timesteps), (x0.size(0), ), dtype=torch.long, device=x0.device)
        xT = self.forward_diffusion_process(x0, noise, t)
        
        rec_sample = self.unet(xT, t).sample
        
        return rec_sample
    
    def get_input(self, batch, num_sampling = None):
        x0, text = batch
        
        if num_sampling is not None:
            x0 = x0[:num_sampling]
            text = text[:num_sampling]
        
        return x0, text
    
    def get_loss(self, batch):
        x0, text = self.get_input(batch)
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        rec_sample = self(x0, noise)
        
        loss = self.criterion(noise, rec_sample)
        
        return loss

    def inference(self, batch, num_sampling, img2img = True):
        self.unet.eval()
        
        with torch.no_grad():
            x0, _ = self.get_input(batch, num_sampling)
            
            if img2img:
                xT = self.forward_diffusion_process(x0)
            else:
                xT = None
            
            pred_x0 = self.reverse_diffusion_process(xT, x0.shape)
        
        self.unet.train()
        
        return pred_x0
    
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
class diffusers_DDIM(nn.Module):
    def __init__(self,
                 optim_name: str,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50,
                 model_path: str = None):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        self.model_path = model_path
        
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
            
    def forward_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        if t is None:
            t = torch.full((x0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=x0.device)
            
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    def reverse_diffusion_process(self, xT = None, shape = None) -> torch.FloatTensor:
        if xT is None:
            xT = torch.randn(shape, dtype=torch.float32).cuda()
        
        pred_x0 = xT
        self.scheduler.set_timesteps(self.num_inference_steps, xT.device)
        for t in self.scheduler.timesteps:
            
            # 1. predict noise model_output
            model_output = self.unet(pred_x0, t).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_x0 = self.scheduler.step(model_output, t, pred_x0).prev_sample
        
        return pred_x0
    
    def forward(self, x0, noise):
        t = torch.randint(0, len(self.scheduler.timesteps), (x0.size(0), ), dtype=torch.long, device=x0.device)
        xT = self.forward_diffusion_process(x0, noise, t)
        
        rec_sample = self.unet(xT, t).sample
        
        return rec_sample
    
    def get_input(self, batch, num_sampling = None):
        x0, text = batch
        
        if num_sampling is not None:
            x0 = x0[:num_sampling]
            text = text[:num_sampling]
        
        return x0, text
    
    def get_loss(self, batch):
        x0, text = self.get_input(batch)
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        rec_sample = self(x0, noise)
        
        loss = self.criterion(noise, rec_sample)
        
        return loss

    def inference(self, batch, num_sampling, img2img = True):
        self.unet.eval()
        
        with torch.no_grad():
            x0, _ = self.get_input(batch, num_sampling)
            
            if img2img:
                xT = self.forward_diffusion_process(x0)
            else:
                xT = None
            
            pred_x0 = self.reverse_diffusion_process(xT, x0.shape)
        
        self.unet.train()
        
        return pred_x0
    
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
class diffusers_PNDM(nn.Module):
    def __init__(self,
                 optim_name: str,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50,
                 model_path: str = None):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        self.model_path = model_path
        
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
            
    def forward_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        if t is None:
            t = torch.full((x0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=x0.device)
            
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    def reverse_diffusion_process(self, xT = None, shape = None) -> torch.FloatTensor:
        if xT is None:
            xT = torch.randn(shape, dtype=torch.float32).cuda()
        
        pred_x0 = xT
        self.scheduler.set_timesteps(self.num_inference_steps, xT.device)
        for t in self.scheduler.timesteps:
            
            # 1. predict noise model_output
            model_output = self.unet(pred_x0, t).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_x0 = self.scheduler.step(model_output, t, pred_x0).prev_sample
        
        return pred_x0
    
    def forward(self, x0, noise):
        t = torch.randint(0, len(self.scheduler._timesteps), (x0.size(0), ), dtype=torch.long, device=x0.device)
        xT = self.forward_diffusion_process(x0, noise, t)
        
        rec_sample = self.unet(xT, t).sample
        
        return rec_sample
    
    def get_input(self, batch, num_sampling = None):
        x0, text = batch
        
        if num_sampling is not None:
            x0 = x0[:num_sampling]
            text = text[:num_sampling]
        
        return x0, text
    
    def get_loss(self, batch):
        x0, text = self.get_input(batch)
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        rec_sample = self(x0, noise)
        
        loss = self.criterion(noise, rec_sample)
        
        return loss
    
    def inference(self, batch, num_sampling, img2img = True):
        self.unet.eval()
        
        with torch.no_grad():
            x0, _ = self.get_input(batch, num_sampling)
            
            if img2img:
                xT = self.forward_diffusion_process(x0)
            else:
                xT = None
            
            pred_x0 = self.reverse_diffusion_process(xT, x0.shape)
        
        self.unet.train()
        
        return pred_x0
      
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
class diffusers_LDM(nn.Module):
    def __init__(self,
                 optim_name: str,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50,
                 model_path: str = None):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        self.model_path = model_path
        
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.unet = instantiate_from_config(unet_config)
        
        self.scheduler = instantiate_from_config(scheduler_config)

        self.model_eval([self.vae])
        
        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
            
    def model_eval(self, models: list):
        for model in models:
            model.requires_grad_(False)
            model.eval()
             
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, zT = None, shape = None) -> torch.FloatTensor:
        if zT is None:
            zT = torch.randn(shape, dtype=torch.float32).cuda()
        
        pred_z0 = zT
        self.scheduler.set_timesteps(self.num_inference_steps, zT.device)
        for t in self.scheduler.timesteps:
            
            # 1. predict noise model_output
            model_output = self.unet(pred_z0, t).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_z0 = self.scheduler.step(model_output, t, pred_z0).prev_sample
        
        return pred_z0
    
    def forward(self, z0, noise):
        t = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zT = self.forward_diffusion_process(z0, noise, t)
        
        rec_sample = self.unet(zT, t).sample
        
        return rec_sample
    
    def get_input(self, batch, num_sampling = None):
        x0, text = batch
        
        if num_sampling is not None:
            x0 = x0[:num_sampling]
            text = text[:num_sampling]
        
        return x0, text
    
    def get_loss(self, batch):
        x0, _ = self.get_input(batch)
        z0 = self.vae.encode(x0).latent_dist.sample()* self.vae.config.scaling_factor
        noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        rec_sample = self(z0, noise)
        
        loss = self.criterion(noise, rec_sample)
        
        return loss
    
    def inference(self, batch, num_sampling, img2img = True):
        self.unet.eval()
        
        with torch.no_grad():
            x0, _ = self.get_input(batch, num_sampling)
            
            if img2img:
                z0 = self.vae.encode(x0).latent_dist.sample()* self.vae.config.scaling_factor
                zT = self.forward_diffusion_process(z0)
            else:
                zT = None
            
            pred_z0 = self.reverse_diffusion_process(zT, x0.shape)
            pred_x0 = self.vae.decode(pred_z0).sample
        
        self.unet.train()
        
        return pred_x0
    
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
class diffusers_StableDiffusion(nn.Module):
    def __init__(self,
                 optim_name: str,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50,
                 model_path: str = None):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        self.model_path = model_path
        
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

        self.model_eval([self.vae, self.text_encoder])
        
        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
            
    def model_eval(self, models: list):
        for model in models:
            model.requires_grad_(False)
            model.eval()
            
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=x0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, zT = None, shape = None, prompt_embeds = None) -> torch.FloatTensor:
        if zT is None:
            zT = torch.randn(shape, dtype=torch.float32).cuda()
        
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(["" * shape[0]])
            
        pred_z0 = zT
        self.scheduler.set_timesteps(self.num_inference_steps, zT.device)
        for t in self.scheduler.timesteps:
            
            # 1. predict noise model_output
            model_output = self.unet(pred_z0, t, prompt_embeds).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_z0 = self.scheduler.step(model_output, t, pred_z0).prev_sample
        
        return pred_z0
    
    def forward(self, z0, noise, prompt_embeds):
        t = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zT = self.forward_diffusion_process(z0, noise, t)
        
        rec_sample = self.unet(zT, t, prompt_embeds).sample
        
        return rec_sample
    
    def get_input(self, batch, num_sampling = None):
        x0, text = batch
        
        if num_sampling is not None:
            x0 = x0[:num_sampling]
            text = text[:num_sampling]
        
        prompt = []
        for t in text:
            prompt.append(str(t.item()))
                
        return x0, prompt
    
    def get_loss(self, batch):
        x0, text = self.get_input(batch)
        
        z0 = self.vae.encode(x0).latent_dist.sample() * self.vae.config.scaling_factor
        prompt_embeds = self.encode_prompt(text)
        noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        rec_sample = self(z0, noise, prompt_embeds)
        
        loss = self.criterion(noise, rec_sample)
        
        return loss
    
    def inference(self, batch, num_sampling, img2img = True):
        self.unet.eval()
        
        with torch.no_grad():
            if img2img:
                x0, text = self.get_input(batch, num_sampling)
                prompt_embeds = self.encode_prompt(text)
                
                z0 = self.vae.encode(x0).latent_dist.sample()* self.vae.config.scaling_factor
                zT = self.forward_diffusion_process(z0)
            else:
                zT, prompt_embeds = None, None
            
            pred_z0 = self.reverse_diffusion_process(zT, x0.shape, prompt_embeds)
            pred_x0 = self.vae.decode(pred_z0).sample
        
        self.unet.train()
        
        return pred_x0
    
    def encode_prompt(self, text):
        with torch.no_grad():
            tokenized_text = self.tokenizer(text,
                                            padding="max_length",
                                            max_length=self.tokenizer.model_max_length,
                                            truncation=True,
                                            return_tensors="pt").input_ids.cuda()
            
            encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state

        return encoder_hidden_states

    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
class diffusers_ControlNet_with_StableDiffusion(nn.Module):
    def __init__(self,
                 optim_name: str,
                 controlnet_config: str,
                 scheduler_config: str,
                 num_inference_steps: int = 50,
                 controlnet_conditioning_scale: float = 1.0,
                 model_path: str = None):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.model_path = model_path
        
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        self.controlnet = instantiate_from_config(controlnet_config)
        
        self.scheduler = instantiate_from_config(scheduler_config)
        
        self.model_eval([self.vae, self.text_encoder, self.unet])
        
        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
            
    def model_eval(self, models: list):
        for model in models:
            model.requires_grad_(False)
            model.eval()
            
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        xT = self.scheduler.add_noise(z0, noise, t)
    
        return xT
    
    def reverse_diffusion_process(self, zT = None, shape = None, prompt_embeds = None, controlnet_cond = None) -> torch.FloatTensor:
        if zT is None:
            zT = torch.randn(shape, dtype=torch.float32).cuda()
        
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(["" * shape[0]])
        
        if controlnet_cond is None:
            controlnet_cond = zT
            
        pred_x0 = zT
        self.scheduler.set_timesteps(self.num_inference_steps, zT.device)
        for t in self.scheduler.timesteps:
            down_block_res_samples, mid_block_res_sample = self.get_controlnet_hidden_blocks(pred_x0, 
                                                                                             t, 
                                                                                             prompt_embeds, 
                                                                                             controlnet_cond)
            
            # 1. predict noise model_output
            model_output = self.unet(pred_x0, t, prompt_embeds,
                                     down_block_additional_residuals=down_block_res_samples,
                                     mid_block_additional_residual=mid_block_res_sample
                                     ).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_x0 = self.scheduler.step(model_output, t, pred_x0).prev_sample
        
        return pred_x0
    
    def forward(self, z0, noise, prompt_embeds, controlnet_cond):
        t = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zT = self.forward_diffusion_process(z0, noise, t)
        down_block_res_samples, mid_block_res_sample = self.get_controlnet_hidden_blocks(zT, t, prompt_embeds, controlnet_cond)
        
        rec_sample = self.unet(zT, t, prompt_embeds,
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample
                               ).sample
        
        return rec_sample
    
    def get_input(self, batch, num_sampling = None):
        x0, text = batch
        
        if num_sampling is not None:
            x0 = x0[:num_sampling]
            text = text[:num_sampling]
        
        prompt = []
        for t in text:
            prompt.append(str(t.item()))
                
        return x0, prompt
    
    def get_loss(self, batch):
        x0, text = self.get_input(batch)
        
        z0 = self.vae.encode(x0).latent_dist.sample() * self.vae.config.scaling_factor
        prompt_embeds = self.encode_prompt(text)
        noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        rec_sample = self(z0, noise, prompt_embeds, x0)
        
        loss = self.criterion(noise, rec_sample)
        
        return loss
    
    def inference(self, batch, num_sampling, img2img = True):
        self.controlnet.eval()
        
        with torch.no_grad():
            if img2img:
                x0, text = self.get_input(batch, num_sampling)
                prompt_embeds = self.encode_prompt(text)
                
                z0 = self.vae.encode(x0).latent_dist.sample()* self.vae.config.scaling_factor
                zT = self.forward_diffusion_process(z0)
            else:
                zT, prompt_embeds = None, None
            
            pred_z0 = self.reverse_diffusion_process(zT, zT.shape, prompt_embeds, x0)
            pred_x0 = self.vae.decode(pred_z0).sample
        
        self.controlnet.train()
        
        return pred_x0
    
    def encode_prompt(self, text):
        with torch.no_grad():
            tokenized_text = self.tokenizer(text,
                                            padding="max_length",
                                            max_length=self.tokenizer.model_max_length,
                                            truncation=True,
                                            return_tensors="pt").input_ids.cuda()
            
            encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state

        return encoder_hidden_states

    def get_controlnet_hidden_blocks(self, zT, t, prompt_embeds, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(zT, t, prompt_embeds, controlnet_cond,
                                                                        return_dict=False)
        down_block_res_samples = [
            down_block_res_sample * self.controlnet_conditioning_scale
            for down_block_res_sample in down_block_res_samples
        ]
        mid_block_res_sample *= self.controlnet_conditioning_scale
        
        return down_block_res_samples, mid_block_res_sample

    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
class frido(Module_base):
    def __init__(self, 
                 optim_target: str,
                 criterion_config: tuple,
                 vae_config: tuple,
                 unet_config: tuple,
                 scheduler_config: tuple,
                 num_inference_steps: int = 50,
                 cloth_warpping: bool = False,
                 cloth_refinement: bool = False,
                 img2img: bool = False,
                 model_path: str = None
                 ):
        super().__init__(model_path)
        self.optimizer = get_obj_from_str(optim_target)
        self.criterion = instantiate_from_config(criterion_config)
        
        self.num_inference_steps = num_inference_steps
        self.cloth_warpping = cloth_warpping
        self.cloth_refinement = cloth_refinement
        self.img2img = img2img
        
        self.tokenizer = CLIPTokenizer.from_pretrained("playgroundai/playground-v2-1024px-aesthetic", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("playgroundai/playground-v2-1024px-aesthetic", subfolder="text_encoder")
        self.vae = instantiate_from_config(vae_config)
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)
        
        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))
    
    def unet_init(self, in_channels):
        # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
        with torch.no_grad():
            # Replace the first conv layer of the unet with a new one with the correct number of input channels
            conv_new = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.unet.conv_in.out_channels,
                kernel_size=3,
                padding=1,
            )

            torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            conv_new.weight.data[:, :9] = self.unet.conv_in.weight.data  # Copy weights from old conv layer
            conv_new.bias.data = self.unet.conv_in.bias.data  # Copy bias from old conv layer

            self.unet.conv_in = conv_new  # replace conv layer in unet
            self.unet.config['in_channels'] = in_channels  # update config
            
    def encode(self, x):
        h_ms = self.vae.encoder(x)

        h_ms = h_ms[::-1]
        prev_h = []
        h_out = []
        for ii in range(len(h_ms)):

            if len(prev_h) != 0:
                for j in range(ii):
                    prev_h[j] = self.vae.upsample[ii-1](prev_h[j])
                    prev_h[j] = self.vae.shared_post_quant_conv[ii-1](prev_h[j])
                
                quant = torch.cat((*prev_h[:ii], h_ms[ii]), dim=1)
                quant = self.vae.shared_decoder[ii-1](quant)
                # quant = quant + prev_h
            else:
                quant = h_ms[ii]

            h = self.vae.ms_quant_conv[ii](quant)
            h_out.append(h)
            quant, _, _ = self.vae.ms_quantize[ii](h)

            prev_h.append(quant)

        h_out = h_out[::-1]
        # # upsample each resolutions
        for i in range(len(h_out)):
            for t in range(i):
                h_out[i] = nn.functional.interpolate(h_out[i], scale_factor=2)
        h_out = h_out[::-1]

        h_out = torch.cat(h_out, dim=1) # channel-wise concate

        return h_out
    
    def decode(self, z):
        h_ms = []
        start = 0
        for i in range(len(self.vae.embed_dim)):
            h_ms.append(z[:, start:start+self.vae.embed_dim[i], :, :])
            start += self.vae.embed_dim[i]

        qaunt_ms = []
        for ii in range(len(h_ms)):
            quant, emb_loss, info = self.vae.ms_quantize[ii](h_ms[ii])
            qaunt_ms.append(quant)

        qaunt_ms = qaunt_ms[::-1]
        quant = torch.cat(qaunt_ms, dim=1) # channel-wise concate

        quant = self.vae.post_quant_conv(quant)
        dec = self.vae.decoder(quant)
        
        return dec
    
    def forward(self):
        pass
    
    def forward_diffusion_process(self, z0, noise = None, t = None):
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        xT = self.scheduler.add_noise(z0, noise, t)
        
        return xT
    
    def reverse_diffusion_process(self):
        pass
    
    def get_input(self, batch, num_sampling = None):
        I = batch["image"]
        captions = batch["captions"]
        
        if num_sampling is not None:
            I = I[:num_sampling]
            captions = captions[:num_sampling]
                
        latent = self.encode(I)
        
        noise = torch.randn(latent.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.encode_prompt(captions)
        
        return latent, noise, encoder_hidden_states
    
    def get_loss(self, batch):
        pass
        
    def inference(self):
        pass
    
    def encode_prompt(self, text):
        with torch.no_grad():
            tokenized_text = self.tokenizer(text,
                                            padding="max_length",
                                            max_length=self.tokenizer.model_max_length,
                                            truncation=True,
                                            return_tensors="pt").input_ids.cuda()
            
            encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state

        return encoder_hidden_states
 
    def get_image_log(self):
        pass
    
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers