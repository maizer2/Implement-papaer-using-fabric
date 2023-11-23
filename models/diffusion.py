import importlib, os
from typing import Any, Optional, Union

import lightning.pytorch as pl
import 
import torch
import torch.nn as nn

from torchvision.utils import make_grid
from torchvision import transforms

from run import get_obj_from_str, instantiate_from_config

from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline

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
            t = torch.full((x0.size(0), ), self.scheduler.timesteps[0], dtype=x0.dtype, device=x0.device)
            
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def reverse_diffusion_process(self, x0 = None, shape = None) -> torch.FloatTensor:
        if x0 is None:
            xT = torch.randn(shape, dtype=torch.float32).cuda()
        else:
            xT = self.forward_diffusion_process(x0)
        
        pred_x0 = xT
        self.scheduler.set_timesteps(self.num_inference_steps, xT.device)
        for t in self.scheduler.timesteps:
            
            # 1. predict noise model_output
            model_output = self.unet(pred_x0, t).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_x0 = self.scheduler.step(model_output, t, pred_x0).prev_sample
        
        return pred_x0
    
    
    def forward(self, x0, noise):
        t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        xT = self.forward_diffusion_process(x0, noise, t)
        
        rec_sample = self.unet(xT, t).sample
        
        return rec_sample
    
    
    def get_loss(self, x0):
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        rec_sample = self(x0, noise)
        
        loss = self.criterion(noise, rec_sample)
        return loss


    def inference(self, x0 = None, shape = None):
        if x0 is not None:
            xT = self.forward_diffusion_process(x0)
        else:
            xT = None
        
        pred_x0 = self.reverse_diffusion_process(xT, shape)
        
        return pred_x0
    

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
    

class Lit_diffusion(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 sampling_step: int,
                 num_sampling: int,
                 img2img: bool,
                 optim_name: str,
                 model_name: str,
                 model_args: tuple) -> None:
        super().__init__()
        self.lr = lr
        self.sampling_step = sampling_step
        self.num_sampling = num_sampling
        self.img2img = img2img
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
    def configure_optimizers(self):
        optim = self.optimizer(self.model.unet.parameters(), self.lr)
        
        return optim
    
    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.current_epoch)
        self.logging_loss(loss, "train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # self.model.set_variable_device(self.device)
        loss = self.model.get_loss(batch, self.current_epoch)
        self.logging_loss(loss, "val")
    
    def test_step(self, batch, batch_idx):
        # self.model.set_variable_device(self.device)
        loss = self.model.get_loss(batch, self.current_epoch)
        self.logging_loss(loss, "test")
    
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
        x0 = batch[0][:self.num_sampling]
        
        if self.img2img:
            pred_x0 = self.model.inference(x0)
        else:
            pred_x0 = self.model.inference(shape=x0.shape)
        
        x0_grid, pred_grid = self.get_grid([x0, pred_x0])
        
        self.logger.experiment.add_image(f'{prefix}/x0', x0_grid, self.current_epoch)
        self.logger.experiment.add_image(f'{prefix}/pred_x0', pred_grid, self.current_epoch)
        
    def logging_output(self, batch, prefix="train"):
        if self.trainer.is_last_batch:
            if self.current_epoch == 0:
                self.sampling(batch, prefix)
            elif (self.current_epoch + 1) % self.sampling_step == 0:
                self.sampling(batch, prefix)