import importlib, os
from typing import Any, Optional, Union

import lightning.pytorch as pl

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import make_grid
from torchvision import transforms

from run import get_obj_from_str, instantiate_from_config

from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, StableDiffusionControlNetPipeline


from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor


class DDPM(nn.Module):
    def __init__(self,
                 unet_config: str,
                 num_train_timesteps = 1_000,
                 num_inference_steps = 1_000):
        super().__init__()
        self.criterion = nn.MSELoss()
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        self.unet = instantiate_from_config(unet_config)
        
        self.beta = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta
    
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
    
class diffusers_DDPM(nn.Module):
    def __init__(self,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 1_000):
        super().__init__()
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

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
    
class diffusers_DDIM(nn.Module):
    def __init__(self,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50):
        super().__init__()
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

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
    
class diffusers_PNDM(nn.Module):
    def __init__(self,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50):
        super().__init__()
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

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
      
class diffusers_LDM(nn.Module):
    def __init__(self,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50):
        super().__init__()
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

        self.model_eval([self.vae])
        
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
    
class diffusers_text_to_LDM(nn.Module):
    def __init__(self,
                 unet_config: str,
                 scheduler_config: str,
                 num_inference_steps = 50):
        super().__init__()
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.unet = instantiate_from_config(unet_config)
        self.scheduler = instantiate_from_config(scheduler_config)

        self.model_eval([self.vae, self.text_encoder])
        
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

class diffusers_ControlNet_with_StableDiffusion(nn.Module):
    def __init__(self,
                 controlnet_config: str,
                 scheduler_config: str,
                 num_inference_steps: int = 50,
                 controlnet_conditioning_scale: float = 1.0):
        super().__init__()
        self.criterion = nn.MSELoss()
        
        self.num_inference_steps = num_inference_steps
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        self.controlnet = instantiate_from_config(controlnet_config)
        
        self.scheduler = instantiate_from_config(scheduler_config)
        
        self.model_eval([self.vae, self.text_encoder, self.unet])
        
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
    
class Lit_diffusion(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 sampling_step: int,
                 num_sampling: int,
                 optim_name: str,
                 model_name: str,
                 model_args: tuple,
                 img2img: bool = True) -> None:
        super().__init__()
        self.lr = lr
        self.sampling_step = sampling_step
        self.num_sampling = num_sampling
        self.img2img = img2img
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
    def configure_optimizers(self):
        optim = self.optimizer(self.model.parameters(), self.lr)
        
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        
        # scheduler = LambdaLR(optim, lr_lambda=[lambda1, lambda2])
        scheduler = LambdaLR(optim, lr_lambda=lambda2)
        
        # return [optim], [scheduler]
        return optim
    
    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch)
        
        self.logging_loss(loss, "train")
        self.logging_output(batch, "train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch)
        
        self.logging_loss(loss, "val")
    
    def test_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch)
        
        self.logging_loss(loss, "test")
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x0_hat = self.predict(batch)
    
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        pass
        
    def predict(self, batch):
        x0_hat = self.model.inference(batch, self.num_sampling, self.img2img)
        
        return x0_hat
        
    def logging_loss(self, loss, prefix):
        self.log(f'{prefix}/loss', loss, prog_bar=True, sync_dist=True)
        
    def get_grid(self, inputs, return_pil=False):        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        outputs = []
        for data in inputs:
            data = (data / 2 + 0.5).clamp(0, 1)
            
            if return_pil:
                outputs.append(self.numpy_to_pil(make_grid(data)))
            else:
                outputs.append(make_grid(data))
        
        return outputs
    
    def sampling(self, batch, prefix="train"):
        x0, _ = self.model.get_input(batch)
        x0_hat = self.predict(batch)
        
        x0_grid, pred_grid = self.get_grid([x0, x0_hat])
        
        self.logger.experiment.add_image(f'{prefix}/x0', x0_grid, self.current_epoch)
        self.logger.experiment.add_image(f'{prefix}/x0_hat', pred_grid, self.current_epoch)
                
    def logging_output(self, batch, prefix="train"):
        if self.global_rank == 0:
            if self.trainer.is_last_batch:
                if self.current_epoch == 0:
                    self.sampling(batch, prefix)
                elif (self.current_epoch + 1) % self.sampling_step == 0:
                    self.sampling(batch, prefix)                