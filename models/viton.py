import importlib, os
from typing import Any, Optional, Union

import lightning.pytorch as pl

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import make_grid
from torchvision import transforms

from loss.vgg import VGGLoss

from run import get_obj_from_str, instantiate_from_config

from diffusers import ControlNetModel, UNet2DModel, AutoencoderKL, StableDiffusionControlNetPipeline


from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor

def model_eval(models: list):
        for model in models:
            model.requires_grad_(False)
            model.eval()
   
class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.from_numpy(self.grid_X, dtype=torch.float32).unsqueeze(0).unsqueeze(3).cuda()
        self.grid_Y = torch.from_numpy(self.grid_Y, dtype=torch.float32).unsqueeze(0).unsqueeze(3).cuda()
        
        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.from_numpy(P_X).cuda()
            P_Y = torch.from_numpy(P_Y).cuda()
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.from_numpy(N,1).fill_(1)
        Z = torch.from_numpy(3,3).fill_(0)       
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L).cuda()
            
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)
        
    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3))
        
        return warped_grid
    
class VITON(nn.Module):
    '''
    CVPR 2018
    VITON: An Image-based Virtual Try-on Network
    https://arxiv.org/abs/1711.08447v4
    '''
    def __init__(self, 
                 stage: str,
                 GC_config: dict,
                 GR_config: dict,
                 GC_model_path: str = None
                 ):
        self.stage = stage
        self.GC = instantiate_from_config(GC_config)
        self.GR = instantiate_from_config(GR_config)
        
        self.Perceptual_loss = VGGLoss()
        self.L1_loss = nn.L1Loss()
        
        if GC_model_path is not None:
            self.GC.load_state_dict(torch.load(GC_model_path))
            model_eval([self.GC])
    
    def stage_1(self, I, M, c, p):
        input = torch.cat([p, c], 1)
        output = self.GC(input)

        I_dot, M_dot = torch.split(output, 3, 1)
        I_dot_loss = self.Perceptual_loss(I_dot, I)
        M_dot_loss = self.L1_loss(M_dot, M)
        
        loss = I_dot_loss + M_dot_loss
        
        return loss
    
    def stage_2(self, I, c_dot, I_dot):
        loss = None
        return loss
    
    def forward(self):
        pass
    
    def get_input(self, batch, num_sampling = None):
        I = batch["image"] # Reference_image
        M = batch["inpaint_mask"] # GT Mask
        
        c = batch["cloth"] # Target_Clothing_image
        
        Pose_map = batch["pose_map"]
        Body_shape = batch["im_pose"]
        Face_and_Hair = batch["im_head"]
        
        p = torch.cat([Pose_map, Body_shape, Face_and_Hair], 1) # Person_Representation
        
        return I, M, c, p
    
    def get_loss(self, batch):
        I, M, c, p = self.get_input(batch)
        
        pass
    
    def inference(self, batch, num_sampling):
        pass
    
class diffusers_VTON_using_StableDiffusion(nn.Module):
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

        model_eval([self.vae, self.text_encoder])
        
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=x0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, zT = None, shape = None, prompt_embeds = None) -> torch.from_numpy:
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

class Lit_viton(pl.LightningModule):
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