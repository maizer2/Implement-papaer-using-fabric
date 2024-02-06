import importlib, os
from typing import Callable, Union, List, Tuple, Dict

import lightning.pytorch as pl

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import make_grid
from torchvision import transforms

from diffusers import UNet2DConditionModel, DDIMScheduler
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from run import instantiate_from_config, get_obj_from_str

from models.base import Module_base

def model_eval(models: list):
    for model in models:
        model.requires_grad_(False)
        model.eval()
        
class cp_vton(Module_base):
    '''
    ECCV 2018
    Toward Characteristic-Preserving Image-based Virtual Try-On Network
    https://arxiv.org/abs/1807.07688
    '''
    def __init__(self, 
                 optim_target: tuple,
                 criterion_config: tuple,
                 stage: str,
                 GMM_config: dict,
                 ploss_config: dict,
                 TOM_config: dict = None,
                 model_path: str = None
                 ):
        super().__init__(optim_target, criterion_config, model_path)
        
        self.stage = stage
        
        if stage == "stage_1":
            self.model, self.GMM = self.get_stage_model(GMM_config)
        else:
            self.model, self.GMM = self.get_stage_model(TOM_config, GMM_config)
            
        self.Perceptual_loss = instantiate_from_config(ploss_config)
        self.L1_loss = nn.L1Loss()
    
    def get_stage_model(self, model_config, GMM_config = None):
        model = instantiate_from_config(model_config)
        
        if self.stage == "stage_1":
            model_GMM = None
        elif self.stage == "stage_2":
            model_GMM = instantiate_from_config(GMM_config)
            model_GMM.load_state_dict(torch.load(self.GMM_model_path))
            model_eval([model_GMM])
        else:
            raise Exception("Wrong stage name.")
        
        return model, model_GMM
    
    def stage_1(self, c, p):
        grid, _ = self.model(p, c)
        c_hat = nn.functional.grid_sample(c, grid, padding_mode='border', align_corners=False)
        
        return c_hat
    
    def stage_2(self, c, p):
        grid, _ = self.GMM(p, c)
        c_hat = nn.functional.grid_sample(c, grid, padding_mode='border', align_corners=False)
        
        input = torch.cat([p, c_hat], 1)
        output = self.model(input)
        p_rendered, m_composite = torch.split(output, 3, 1)
        
        p_rendered = nn.functional.tanh(p_rendered)
        m_composite = nn.functional.sigmoid(m_composite)
        
        I_O = c_hat * m_composite + p_rendered * (1 - m_composite)
        
        return I_O
    
    def forward(self):
        pass
    
    def get_input(self, batch, num_sampling = None):
        I_t = batch["image"] # Reference_image
        c_t = batch["im_cloth"] # Clothes on Person
        
        c = batch["cloth"] # In-shop Clothes
        
        Pose_map = batch["pose_map"]
        Body_shape = batch["im_pose"]
        Face_and_Hair = batch["im_head"]
        
        p = torch.cat([Pose_map, Body_shape, Face_and_Hair], 1) # Person_Representation
        
        if num_sampling is not None:
            I_t, c_t, c, p = I_t[:num_sampling], c_t[:num_sampling], c[:num_sampling], p[:num_sampling]
            
        return I_t, c_t, c, p
    
    def get_loss(self, batch):
        I_t, c_t, c, p = self.get_input(batch)
        
        if self.stage == "stage_1":
            c_hat = self.stage_1(c, p)
            loss = self.L1_loss(c_hat, c_t)
        elif self.stage == "stage_2":
            I_O = self.stage_2(c, p)
            loss = self.Perceptual_loss(I_O, I_t) + self.L1_loss(I_O, I_t)
        else:
            raise Exception("Wrong stage.")
        
        return [("total", loss)]
    
    def inference(self, batch, num_sampling, model_save=False):
        I_t, c_t, c, p = self.get_input(batch, num_sampling)
        
        self.model.eval()
        
        with torch.no_grad():
            if self.stage == "stage_1":
                if model_save:
                    torch.save(self.model.state_dict(), self.GMM_model_path)
                real = c_t
                fake = self.stage_1(c, p)
            elif self.stage == "stage_2":
                real = I_t
                fake = self.stage_2(c, p)
            else:
                raise Exception("Wrong stage.")
        
        self.model.train()
        
        return real, fake
    
    def save_warped_image(self, batch):
        os.makedirs("warped_cloths_paired", exist_ok=True)
        _, _, c, p = self.get_input(batch)
        cloth_name = batch["c_name"]
        
        self.model.load_state_dict(torch.load(self.GMM_model_path))
        model_eval([self.model])
            
        with torch.no_grad():
            warped_cloth = self.stage_1(c, p)
            warped_cloth = (warped_cloth / 2 + 0.5).clamp(0, 1)
            
        for image, image_name in zip(warped_cloth, cloth_name):
            transforms.ToPILImage()(image).save(os.path.join("warped_cloths_paired", image_name))

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
    
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.model.parameters(), lr)
        
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lr_lambda=lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers

    def predict(self, batch, save_dir):
        
        topil = transforms.ToPILImage()
        
        outputs = self.inference(batch)
        
        
        real = (outputs["real"] / 2 + 0.5).clamp(0, 1)
        fake = (outputs["fake"] / 2 + 0.5).clamp(0, 1)
        
        real_pils = [topil(real_pt) for real_pt in real]
        fake_pils = [topil(fake_pt) for fake_pt in fake]
        
        for idx in range(len(real_pils)):
            out_dir = os.path.join(save_dir, "inference", outputs["fake_id"][idx].split('.')[0])
            os.makedirs(out_dir, exist_ok=True)
            
            real_pils[idx].save(os.path.join(out_dir, "real.png"))
            fake_pils[idx].save(os.path.join(out_dir, "fake.png"))
     
class stable_diffusion_text_guided_inpainting(Module_base):
    def __init__(self,
                 optim_target: tuple,
                 criterion_config: tuple,
                 num_inference_steps: int = 50,
                 use_caption: bool = False,
                 model_path = None, # .../unet.ckpt
                 train_resume: bool = False
                 ):        
        super().__init__(optim_target, criterion_config, model_path, train_resume)

        self.num_inference_steps = num_inference_steps
        self.use_caption = use_caption
                
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
        
        eval_models = [self.text_encoder, self.vae]
        model_eval(eval_models)

        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline as pipeline
        self.pipeline = pipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
        
    def training_resume(self):
        # This method is executed by the pl.LightningModule.
        if self.train_resume and os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            print("Unet loaded")
            self.unet.load_state_dict(torch.load(self.model_path))
         
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self):
        # Not using
        pass
    
    def get_loss(self, batch):
        z0, unet_input, noise, encoder_hidden_states = self.pre_process(batch)
        unet_output = self(z0, unet_input, noise, encoder_hidden_states)
        loss = self.criterion(noise, unet_output)
        
        return {"total": loss}
    
    def pre_process(self, batch, num_sampling = None, inference = False):
        x0, i_m, m, text, _, _ = self.get_input(batch, num_sampling)
        
        down_m      = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        z0          = self.vae.encode(x0).latent_dist.sample() * self.vae.config.scaling_factor
        latent_i_m  = self.vae.encode(i_m).latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input  = torch.cat([down_m, latent_i_m], 1)
        noise       = torch.randn(z0.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.get_encoder_hidden_states(text=text)
                
        return z0, unet_input, noise, encoder_hidden_states
    
    def forward(self, z0, unet_input, noise, encoder_hidden_states):
        timestep = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zt = self.forward_diffusion_process(z0, noise, timestep)
        sample = torch.cat([zt, unet_input], 1)
        
        rec_sample = self.unet(sample=sample, 
                               timestep=timestep, 
                               encoder_hidden_states=encoder_hidden_states).sample
        
        return rec_sample
    
    def inference(self, batch, num_sampling=None):
        image, _, mask_image, prompt, im_name, c_name = self.get_input(batch, num_sampling)
        
        self.pipeline.to("cuda")
        x0_pred = self.pipeline(prompt=prompt,
                                image=image,
                                mask_image=mask_image,
                                height = 512,
                                width = 384,
                                num_inference_steps=self.num_inference_steps,
                                output_type="pt"
                                ).images
        # re normalize
        ## because pipeline has denormalize
        x0_pred = 2.0 * x0_pred - 1.0
        
        return {"real": image, 
                "fake": x0_pred,
                "image_id": im_name,
                "cloth_id": c_name}
    
    def get_input(self, batch, num_sampling = None):
        i = batch["image"]
        i_m = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        text = self.get_text_encoder_input(batch["captions"], batch["category"])
        
        im_name = batch["im_name"]
        c_name = batch["c_name"]
        
        if num_sampling is not None:
            i = i[:num_sampling]
            i_m = i_m[:num_sampling]
            m = m[:num_sampling]
            text = text[:num_sampling]
            
            im_name = im_name[:num_sampling]
            c_name = c_name[:num_sampling]
            
        return i, i_m, m, text, im_name, c_name
        
    def get_text_encoder_input(self, caption, category):
        if self.use_caption:
            text = [cap for
                    cap in caption]
        else:
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',
            }
            
            # batch size lenght
            text = [f'a photo of a model wearing {category_text[ctg]}' for
                    ctg in category]
        
        return text
    
    def get_tokenized_text(self, text):        
        # Tokenize text ( bsz, 77 )
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.cuda()
        
        return tokenized_text
    
    def get_encoder_hidden_states(self, text = None):
        if text is None:
            text = ""
            
        tokenized_text = self.get_tokenized_text(text)
        encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state
            
        return encoder_hidden_states
     
    def get_image_log(self, batch, num_sampling):
        outputs = self.inference(batch, num_sampling)

        return {"real": outputs["real"],
                "fake": outputs["fake"]}
        
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.unet.parameters(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    
    def predict(self, batch, save_dir):
        
        topil = transforms.ToPILImage()
        
        outputs = self.inference(batch)
        
        real = (outputs["real"] / 2 + 0.5).clamp(0, 1)
        fake = (outputs["fake"] / 2 + 0.5).clamp(0, 1)
        
        real_pils = [topil(real_pt) for real_pt in real]
        fake_pils = [topil(fake_pt) for fake_pt in fake]
        
        for idx in range(len(real_pils)):
            fake_id = outputs['image_id'][idx].split('.')[0] + "_" + outputs['cloth_id'][idx].split('.')[0]
            
            if outputs['image_id'] == outputs['cloth_id']:
                out_dir = os.path.join(save_dir, "inference", "paired", fake_id)
            else:
                out_dir = os.path.join(save_dir, "inference", "unpaired", fake_id)
                
            os.makedirs(out_dir, exist_ok=True)
            
            real_pils[idx].save(os.path.join(out_dir, "real.png"))
            fake_pils[idx].save(os.path.join(out_dir, "fake.png"))
            
# ToDo
class stable_diffusion_text_guided_inpainting_with_controlnet(Module_base):
    def __init__(self,
                 optim_target: tuple,
                 criterion_config: tuple,
                 num_inference_steps: int = 50,
                 use_caption: bool = False,
                 use_cloth_warpping: bool = True,
                 use_cloth_refinement: bool = False,
                 conditioning_scale = 1.0,
                 model_path = None, # .../unet.ckpt
                 train_resume: bool = False
                 ):        
        super().__init__(optim_target, criterion_config, model_path, train_resume)

        self.num_inference_steps = num_inference_steps
        self.use_caption = use_caption
        self.use_cloth_warpping = use_cloth_warpping
        self.use_cloth_refinement = use_cloth_refinement
        self.conditioning_scale = conditioning_scale
                
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
        self.controlnet = ControlNetModel.from_unet(unet=self.unet,
                                                    conditioning_embedding_out_channels=[16, 32, 96, 256])
        
        eval_models = [self.text_encoder, self.vae, self.unet]
        model_eval(eval_models)

        from models.pipeline.pipeline_stable_diffusion_inpaint_with_controlnet import StableDiffusionInpaintWithControlnetPipeline as pipeline
        self.pipeline = pipeline(vae=self.vae,
                                 text_encoder=self.text_encoder,
                                 tokenizer=self.tokenizer,
                                 unet=self.unet,
                                 scheduler=self.scheduler,
                                 controlnet=self.controlnet)
        
    def training_resume(self):
        # This method is executed by the pl.LightningModule.
        if self.train_resume and os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            print("Unet loaded")
            self.unet.load_state_dict(torch.load(self.model_path))
         
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self):
        # Not using
        pass
    
    def get_loss(self, batch):
        z0, unet_input, noise, encoder_hidden_states, controlnet_cond = self.pre_process(batch)
        unet_output = self(z0, unet_input, noise, encoder_hidden_states, controlnet_cond)
        loss = self.criterion(noise, unet_output)
        
        return {"total": loss}
    
    def pre_process(self, batch, num_sampling = None):
        x0, c, c_w, i_m, m, p, text, _, _ = self.get_input(batch, num_sampling)
        
        down_m      = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        z0          = self.vae.encode(x0).latent_dist.sample() * self.vae.config.scaling_factor
        latent_i_m  = self.vae.encode(i_m).latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input  = torch.cat([down_m, latent_i_m], 1)
        noise       = torch.randn(z0.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.get_encoder_hidden_states(text=text)
        controlnet_cond = c_w
        
        return z0, unet_input, noise, encoder_hidden_states, controlnet_cond
    
    def forward(self, z0, unet_input, noise, encoder_hidden_states, controlnet_cond):
        timestep = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zt = self.forward_diffusion_process(z0, noise, timestep)
        sample = torch.cat([zt, unet_input], 1)
        down_block_res_samples, mid_block_res_sample = self.get_controlnet_hidden_blocks(sample, timestep, encoder_hidden_states, controlnet_cond)
        
        rec_sample = self.unet(sample=sample, 
                               timestep=timestep, 
                               encoder_hidden_states=encoder_hidden_states,
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample).sample
        
        return rec_sample
    
    def inference(self, batch, num_sampling=None):
        image, cloth, warped_cloth_image, masked_image, mask_image, prompt, im_name, c_name = self.get_input(batch, num_sampling)
        
        self.pipeline.to("cuda")
        x0_pred = self.pipeline(prompt=prompt,
                                image=image,
                                mask_image=mask_image,
                                height = 512,
                                width = 384,
                                num_inference_steps=self.num_inference_steps,
                                output_type="pt"
                                ).images
        # re normalize
        ## because pipeline has denormalize
        x0_pred = 2.0 * x0_pred - 1.0
        
        return {"real": image, 
                "fake": x0_pred,
                "cloth": cloth,
                "warped_cloth": warped_cloth_image,
                "image_id": im_name,
                "cloth_id": c_name}
    
    def get_input(self, batch, num_sampling = None):
        i = batch["image"]
        c = batch["cloth"]
        i_m = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        p = batch["pose_map"]
        c_w = batch.get("warped_cloth") if batch.get("warped_cloth") is not None else self.get_warped_cloth(c, i_m, p)
        text = self.get_text_encoder_input(batch["captions"], batch["category"])
        
        im_name = batch["im_name"]
        c_name = batch["c_name"]
        
        if num_sampling is not None:
            i = i[:num_sampling]
            c = c[:num_sampling]
            i_m = i_m[:num_sampling]
            m = m[:num_sampling]
            p = p[:num_sampling]
            c_w = c_w[:num_sampling]
            text = text[:num_sampling]
            
            im_name = im_name[:num_sampling]
            c_name = c_name[:num_sampling]
            
        return i, c, c_w, i_m, m, p, text, im_name, c_name
    
    def warpping_cloth(self, cloth, im_mask, pose_map):
        # TPS parameters prediction
        # For sake of performance, the TPS parameters are predicted on a low resolution image
        low_cloth = transforms.functional.resize(cloth, (256, 192),
                                                 transforms.InterpolationMode.BILINEAR,
                                                 antialias=True)
        low_im_mask = transforms.functional.resize(im_mask, (256, 192),
                                                   transforms.InterpolationMode.BILINEAR,
                                                   antialias=True)
        low_pose_map = transforms.functional.resize(pose_map, (256, 192),
                                                    transforms.InterpolationMode.BILINEAR,
                                                    antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, _, _, _, _, _, _, _ = self.tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(cloth.size(2), cloth.size(3)), 
                               interpolation=transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)

        warped_cloth = nn.functional.grid_sample(cloth, highres_grid, padding_mode='border', align_corners=True)
        
        if self.use_cloth_refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth)
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth
        
        return warped_cloth
    
    def get_warped_cloth(self, cloth, im_mask, pose_map):
        if self.use_cloth_warpping:
            warped_cloth = self.warpping_cloth(cloth, im_mask, pose_map)
        else:
            warped_cloth = cloth
        
        return warped_cloth
    
    def get_text_encoder_input(self, caption, category):
        if self.use_caption:
            text = [cap for
                    cap in caption]
        else:
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',
            }
            
            # batch size lenght
            text = [f'a photo of a model wearing {category_text[ctg]}' for
                    ctg in category]
        
        return text
    
    def get_tokenized_text(self, text):        
        # Tokenize text ( bsz, 77 )
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.cuda()
        
        return tokenized_text
    
    def get_encoder_hidden_states(self, text = None):
        if text is None:
            text = ""
            
        tokenized_text = self.get_tokenized_text(text)
        encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state
            
        return encoder_hidden_states
     
    def get_controlnet_hidden_blocks(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(sample=sample,
                                                                       timestep=timestep,
                                                                       encoder_hidden_states=encoder_hidden_states,
                                                                       controlnet_cond=controlnet_cond,
                                                                       conditioning_scale=self.conditioning_scale,
                                                                       return_dict=False)
        
        return down_block_res_samples, mid_block_res_sample
        
    def get_image_log(self, batch, num_sampling):
        outputs = self.inference(batch, num_sampling)

        return {"real": outputs["real"],
                "fake": outputs["fake"]}
            
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.unet.parameters(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
 
    def predict(self, batch, save_dir):
        
        topil = transforms.ToPILImage()
        
        outputs = self.inference(batch)
        
        real = (outputs["real"] / 2 + 0.5).clamp(0, 1)
        fake = (outputs["fake"] / 2 + 0.5).clamp(0, 1)
        cloth = (outputs["cloth"] / 2 + 0.5).clamp(0, 1)
        w_cloth = (outputs["warped_cloth"] / 2 + 0.5).clamp(0, 1)
        
        real_pils = [topil(real_pt) for real_pt in real]
        fake_pils = [topil(fake_pt) for fake_pt in fake]
        cloth_pils = [topil(cloth_pt) for cloth_pt in cloth]
        w_cloth_pils = [topil(w_cloth_pt) for w_cloth_pt in w_cloth]
        
        for idx in range(len(real_pils)):
            fake_id = outputs['image_id'][idx].split('.')[0] + "_" + outputs['cloth_id'][idx].split('.')[0]
            
            if outputs['image_id'] == outputs['cloth_id']:
                out_dir = os.path.join(save_dir, "inference", "paired", fake_id)
            else:
                out_dir = os.path.join(save_dir, "inference", "unpaired", fake_id)
                
            os.makedirs(out_dir, exist_ok=True)
            
            real_pils[idx].save(os.path.join(out_dir, "real.png"))
            fake_pils[idx].save(os.path.join(out_dir, "fake.png"))
            cloth_pils[idx].save(os.path.join(out_dir, "cloth.png"))
            w_cloth_pils[idx].save(os.path.join(out_dir, "warped_cloth.png"))
     
class stable_diffusion_text_guided_inpainting_vton(Module_base):
    def __init__(self,
                 optim_target: tuple,
                 criterion_config: tuple,
                 dataset_name: str = "vitonhd", # ["vitonhd", "dresscode"]
                 in_channels: int = 31,
                 num_inference_steps: int = 50,
                 use_caption: bool = False,
                 use_cloth_warpping: bool = True,
                 use_cloth_refinement: bool = False,
                 use_img2img: bool = True,
                 model_path = None, # .../unet.ckpt
                 train_resume: bool = False
                 ):        
        super().__init__(optim_target, criterion_config, model_path, train_resume)

        self.dataset_name = dataset_name
        self.in_channels = in_channels
        self.num_inference_steps = num_inference_steps
        self.use_caption = use_caption
        self.use_cloth_warpping = use_cloth_warpping
        self.use_cloth_refinement = use_cloth_refinement
        self.use_img2img = use_img2img
        
        # refinemnet network casuses the garment pattern to disappear.
        # I recommend not using refinemnet network.
        # If you want to use refinement network, please specify option "use_cloth_refinemnet" in the yaml file.
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module',
                                                   dataset=dataset_name)
        
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
        self.unet_new_in_channels()
        
        eval_models = [self.tps, self.refinement, self.text_encoder, self.vae]
        model_eval(eval_models)

        from models.pipeline.pipeline_stable_diffusion_inpaint_vton import StableDiffusionInpaintVtonPipeline as pipeline
        self.pipeline = pipeline(vae=self.vae,
                                 text_encoder=self.text_encoder,
                                 tokenizer=self.tokenizer,
                                 unet=self.unet,
                                 scheduler=self.scheduler,
                                 tps=self.tps,
                                 refinement=self.refinement)
        
    def unet_new_in_channels(self):
        if self.train_resume:
            if self.unet.config.in_channels != self.unet.conv_in.in_channels:
                raise ValueError(f"unet conv_in channels[{self.unet.conv_in.in_channels}] and unet config in_channels[{self.unet.config.in_channels}] is different.")
        else:
            # 9channels -> 31channels
            # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
            with torch.no_grad():
                # Replace the first conv layer of the unet with a new one with the correct number of input channels
                conv_new = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.unet.conv_in.out_channels,
                    kernel_size=3,
                    padding=1,
                )

            torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            conv_new.weight.data[:, :9] = self.unet.conv_in.weight.data  # Copy weights from old conv layer
            conv_new.bias.data = self.unet.conv_in.bias.data  # Copy bias from old conv layer
            
            # replace conv layer in unet
            self.unet.conv_in = conv_new  
             # update config
            self.unet.config.in_channels = self.in_channels 
            self.unet.config["in_channels"] = self.in_channels
    
    def training_resume(self):
        # This method is executed by the pl.LightningModule.
        if self.train_resume and os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            print("Unet loaded")
            self.unet.load_state_dict(torch.load(self.model_path))
         
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, latent_shape, encoder_hidden_states, zT = None, unet_input = None):
        # Not using
        pass
    
    def get_loss(self, batch):
        z0, unet_input, noise, encoder_hidden_states = self.pre_process(batch)
        unet_output = self(z0, unet_input, noise, encoder_hidden_states)
        loss = self.criterion(noise, unet_output)
        
        return {"total": loss}
    
    def pre_process(self, batch, num_sampling = None, inference = False):
        x0, c, c_w, i_m, m, p, text, _, _ = self.get_input(batch, num_sampling)
        
        down_m      = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        down_p      = nn.functional.interpolate(p, size=(p.shape[2] // 8, p.shape[3] // 8), mode="bilinear")
        z0          = self.vae.encode(x0).latent_dist.sample() * self.vae.config.scaling_factor
        latent_c_w  = self.vae.encode(c_w).latent_dist.sample() * self.vae.config.scaling_factor
        latent_i_m  = self.vae.encode(i_m).latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input  = torch.cat([down_m, down_p, latent_c_w, latent_i_m], 1)
        noise       = torch.randn(z0.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.get_encoder_hidden_states(text=text)
        
        if inference:
            pass
        
        return z0, unet_input, noise, encoder_hidden_states
    
    def forward(self, z0, unet_input, noise, encoder_hidden_states):
        timestep = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zt = self.forward_diffusion_process(z0, noise, timestep)
        sample = torch.cat([zt, unet_input], 1)
        
        rec_sample = self.unet(sample=sample, 
                               timestep=timestep, 
                               encoder_hidden_states=encoder_hidden_states).sample
        
        return rec_sample
    
    def inference(self, batch, num_sampling=None):
        image, cloth, warped_cloth_image, masked_image, mask_image, posemap, prompt, im_name, c_name = self.get_input(batch, num_sampling)
        
        self.pipeline.to("cuda")
        x0_pred = self.pipeline(prompt=prompt,
                                image=image,
                                mask_image=mask_image,
                                cloth_image=cloth,
                                warped_cloth_image=warped_cloth_image,
                                posemap_image=posemap,
                                height = 512,
                                width = 384,
                                num_inference_steps=self.num_inference_steps,
                                output_type="pt",
                                use_cloth_warpping=self.use_cloth_warpping,
                                use_cloth_refinemnet=self.use_cloth_refinement
                                ).images
        # re normalize
        ## because pipeline has denormalize
        x0_pred = 2.0 * x0_pred - 1.0
            
        return {"real": image, 
                "fake": x0_pred,
                "cloth": cloth,
                "warped_cloth": warped_cloth_image,
                "image_id": im_name,
                "cloth_id": c_name}
    
    def get_input(self, batch, num_sampling = None):
        i = batch["image"]
        c = batch["cloth"]
        i_m = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        p = batch["pose_map"]
        c_w = batch.get("warped_cloth") if batch.get("warped_cloth") is not None else self.get_warped_cloth(c, i_m, p)
        text = self.get_text_encoder_input(batch["captions"], batch["category"])
        
        im_name = batch["im_name"]
        c_name = batch["c_name"]
        
        if num_sampling is not None:
            i = i[:num_sampling]
            c = c[:num_sampling]
            i_m = i_m[:num_sampling]
            m = m[:num_sampling]
            p = p[:num_sampling]
            c_w = c_w[:num_sampling]
            text = text[:num_sampling]
            
            im_name = im_name[:num_sampling]
            c_name = c_name[:num_sampling]
            
        return i, c, c_w, i_m, m, p, text, im_name, c_name
        
    def warpping_cloth(self, cloth, im_mask, pose_map):
        # TPS parameters prediction
        # For sake of performance, the TPS parameters are predicted on a low resolution image
        low_cloth = transforms.functional.resize(cloth, (256, 192),
                                                 transforms.InterpolationMode.BILINEAR,
                                                 antialias=True)
        low_im_mask = transforms.functional.resize(im_mask, (256, 192),
                                                   transforms.InterpolationMode.BILINEAR,
                                                   antialias=True)
        low_pose_map = transforms.functional.resize(pose_map, (256, 192),
                                                    transforms.InterpolationMode.BILINEAR,
                                                    antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, _, _, _, _, _, _, _ = self.tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(cloth.size(2), cloth.size(3)), 
                               interpolation=transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)

        warped_cloth = nn.functional.grid_sample(cloth, highres_grid, padding_mode='border', align_corners=True)
        
        if self.use_cloth_refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth)
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth
        
        return warped_cloth
    
    def get_warped_cloth(self, cloth, im_mask, pose_map):
        if self.use_cloth_warpping:
            warped_cloth = self.warpping_cloth(cloth, im_mask, pose_map)
        else:
            warped_cloth = cloth
        
        return warped_cloth
    
    def get_text_encoder_input(self, caption, category):
        if self.use_caption:
            text = [cap for
                    cap in caption]
        else:
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',
            }
            
            # batch size lenght
            text = [f'a photo of a model wearing {category_text[ctg]}' for
                    ctg in category]
        
        return text
    
    def get_tokenized_text(self, text):        
        # Tokenize text ( bsz, 77 )
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.cuda()
        
        return tokenized_text
    
    def get_encoder_hidden_states(self, text = None):
        if text is None:
            text = ""
            
        tokenized_text = self.get_tokenized_text(text)
        encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state
            
        return encoder_hidden_states
     
    def get_image_log(self, batch, num_sampling):
        outputs = self.inference(batch, num_sampling)

        return {"real": outputs["real"],
                "fake": outputs["fake"]}
        
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.unet.parameters(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers

    def predict(self, batch, save_dir):
        
        topil = transforms.ToPILImage()
        
        outputs = self.inference(batch)
        
        real = (outputs["real"] / 2 + 0.5).clamp(0, 1)
        fake = (outputs["fake"] / 2 + 0.5).clamp(0, 1)
        cloth = (outputs["cloth"] / 2 + 0.5).clamp(0, 1)
        w_cloth = (outputs["warped_cloth"] / 2 + 0.5).clamp(0, 1)
        
        real_pils = [topil(real_pt) for real_pt in real]
        fake_pils = [topil(fake_pt) for fake_pt in fake]
        cloth_pils = [topil(cloth_pt) for cloth_pt in cloth]
        w_cloth_pils = [topil(w_cloth_pt) for w_cloth_pt in w_cloth]
        
        for idx in range(len(real_pils)):
            fake_id = outputs['image_id'][idx].split('.')[0] + "_" + outputs['cloth_id'][idx].split('.')[0]
            
            if outputs['image_id'] == outputs['cloth_id']:
                out_dir = os.path.join(save_dir, "inference", "paired", fake_id)
            else:
                out_dir = os.path.join(save_dir, "inference", "unpaired", fake_id)
                
            os.makedirs(out_dir, exist_ok=True)
            
            real_pils[idx].save(os.path.join(out_dir, "real.png"))
            fake_pils[idx].save(os.path.join(out_dir, "fake.png"))
            cloth_pils[idx].save(os.path.join(out_dir, "cloth.png"))
            w_cloth_pils[idx].save(os.path.join(out_dir, "warped_cloth.png"))
            
# ToDo
class stable_diffusion_text_guided_inpainting_vton_with_controlnet(Module_base):
    def __init__(self,
                 optim_target: tuple,
                 criterion_config: tuple,
                 dataset_name: str = "vitonhd", # ["vitonhd", "dresscode"]
                 in_channels: int = 31,
                 num_inference_steps: int = 50,
                 use_caption: bool = False,
                 use_cloth_warpping: bool = True,
                 use_cloth_refinement: bool = False,
                 use_img2img: bool = True,
                 conditioning_scale = 1.0,
                 model_path = None, # .../unet.ckpt
                 train_resume: bool = False
                 ):        
        super().__init__(optim_target, criterion_config, model_path, train_resume)

        self.dataset_name = dataset_name
        self.in_channels = in_channels
        self.num_inference_steps = num_inference_steps
        self.use_caption = use_caption
        self.use_cloth_warpping = use_cloth_warpping
        self.use_cloth_refinement = use_cloth_refinement
        self.use_img2img = use_img2img
        self.conditioning_scale = conditioning_scale
        
        # refinemnet network casuses the garment pattern to disappear.
        # I recommend not using refinemnet network.
        # If you want to use refinement network, please specify option "use_cloth_refinemnet" in the yaml file.
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module',
                                                   dataset=dataset_name)
        
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
        self.unet_new_in_channels()
        self.controlnet = ControlNetModel.from_unet(unet=self.unet,
                                                    conditioning_embedding_out_channels=[16, 32, 96, 256])
        
        eval_models = [self.tps, self.refinement, self.text_encoder, self.vae, self.unet]
        model_eval(eval_models)

        from models.pipeline.pipeline_stable_diffusion_inpaint_vton_with_controlnet import StableDiffusionInpaintVtonWithControlnetPipeline as pipeline
        self.pipeline = pipeline(vae=self.vae,
                                 text_encoder=self.text_encoder,
                                 tokenizer=self.tokenizer,
                                 unet=self.unet,
                                 scheduler=self.scheduler,
                                 tps=self.tps,
                                 refinement=self.refinement,
                                 controlnet=self.controlnet)
        
    def unet_new_in_channels(self):
        if self.train_resume:
            if self.unet.config.in_channels != self.unet.conv_in.in_channels:
                raise ValueError(f"unet conv_in channels[{self.unet.conv_in.in_channels}] and unet config in_channels[{self.unet.config.in_channels}] is different.")
        else:
            # 9channels -> 31channels
            # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
            with torch.no_grad():
                # Replace the first conv layer of the unet with a new one with the correct number of input channels
                conv_new = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.unet.conv_in.out_channels,
                    kernel_size=3,
                    padding=1,
                )

            torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            conv_new.weight.data[:, :9] = self.unet.conv_in.weight.data  # Copy weights from old conv layer
            conv_new.bias.data = self.unet.conv_in.bias.data  # Copy bias from old conv layer
            
            # replace conv layer in unet
            self.unet.conv_in = conv_new  
             # update config
            self.unet.config.in_channels = self.in_channels 
            self.unet.config["in_channels"] = self.in_channels
    
    def training_resume(self):
        # This method is executed by the pl.LightningModule.
        if self.train_resume and os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            print("Unet loaded")
            self.unet.load_state_dict(torch.load(self.model_path))
         
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, latent_shape, encoder_hidden_states, zT = None, unet_input = None):
        # Not using
        pass
    
    def get_loss(self, batch):
        z0, unet_input, noise, encoder_hidden_states, controlnet_cond = self.pre_process(batch)
        unet_output = self(z0, unet_input, noise, encoder_hidden_states, controlnet_cond)
        loss = self.criterion(noise, unet_output)
        
        return {"total": loss}
    
    def pre_process(self, batch, num_sampling = None, inference = False):
        x0, c, c_w, i_m, m, p, text, _, _ = self.get_input(batch, num_sampling)
        
        down_m      = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        down_p      = nn.functional.interpolate(p, size=(p.shape[2] // 8, p.shape[3] // 8), mode="bilinear")
        z0          = self.vae.encode(x0).latent_dist.sample() * self.vae.config.scaling_factor
        latent_c_w  = self.vae.encode(c_w).latent_dist.sample() * self.vae.config.scaling_factor
        latent_i_m  = self.vae.encode(i_m).latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input  = torch.cat([down_m, down_p, latent_c_w, latent_i_m], 1)
        noise       = torch.randn(z0.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.get_encoder_hidden_states(text=text)
        controlnet_cond = c_w
        
        if inference:
            pass
        
        return z0, unet_input, noise, encoder_hidden_states, controlnet_cond
    
    def forward(self, z0, unet_input, noise, encoder_hidden_states, controlnet_cond):
        timestep = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zt = self.forward_diffusion_process(z0, noise, timestep)
        sample = torch.cat([zt, unet_input], 1)
        down_block_res_samples, mid_block_res_sample = self.get_controlnet_hidden_blocks(sample, timestep, encoder_hidden_states, controlnet_cond)
        
        rec_sample = self.unet(sample=sample, 
                               timestep=timestep, 
                               encoder_hidden_states=encoder_hidden_states,
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample).sample
        
        return rec_sample
    
    def inference(self, batch, num_sampling=None):
        image, cloth, warped_cloth_image, masked_image, mask_image, posemap, prompt, im_name, c_name = self.get_input(batch, num_sampling)
        
        self.pipeline.to("cuda")
        x0_pred = self.pipeline(prompt=prompt,
                                image=image,
                                mask_image=mask_image,
                                cloth_image=cloth,
                                warped_cloth_image=warped_cloth_image,
                                posemap_image=posemap,
                                height = 512,
                                width = 384,
                                num_inference_steps=self.num_inference_steps,
                                output_type="pt",
                                use_cloth_warpping=self.use_cloth_warpping,
                                use_cloth_refinemnet=self.use_cloth_refinement
                                ).images
        # re normalize
        ## because pipeline has denormalize
        x0_pred = 2.0 * x0_pred - 1.0
            
        return {"real": image, 
                "fake": x0_pred,
                "cloth": cloth,
                "warped_cloth": warped_cloth_image,
                "image_id": im_name,
                "cloth_id": c_name}
    
    def get_input(self, batch, num_sampling = None):
        i = batch["image"]
        c = batch["cloth"]
        i_m = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        p = batch["pose_map"]
        c_w = batch.get("warped_cloth") if batch.get("warped_cloth") is not None else self.get_warped_cloth(c, i_m, p)
        text = self.get_text_encoder_input(batch["captions"], batch["category"])
        
        im_name = batch["im_name"]
        c_name = batch["c_name"]
        
        if num_sampling is not None:
            i = i[:num_sampling]
            c = c[:num_sampling]
            i_m = i_m[:num_sampling]
            m = m[:num_sampling]
            p = p[:num_sampling]
            c_w = c_w[:num_sampling]
            text = text[:num_sampling]
            
            im_name = im_name[:num_sampling]
            c_name = c_name[:num_sampling]
            
        return i, c, c_w, i_m, m, p, text, im_name, c_name
        
    def warpping_cloth(self, cloth, im_mask, pose_map):
        # TPS parameters prediction
        # For sake of performance, the TPS parameters are predicted on a low resolution image
        low_cloth = transforms.functional.resize(cloth, (256, 192),
                                                 transforms.InterpolationMode.BILINEAR,
                                                 antialias=True)
        low_im_mask = transforms.functional.resize(im_mask, (256, 192),
                                                   transforms.InterpolationMode.BILINEAR,
                                                   antialias=True)
        low_pose_map = transforms.functional.resize(pose_map, (256, 192),
                                                    transforms.InterpolationMode.BILINEAR,
                                                    antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, _, _, _, _, _, _, _ = self.tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(cloth.size(2), cloth.size(3)), 
                               interpolation=transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)

        warped_cloth = nn.functional.grid_sample(cloth, highres_grid, padding_mode='border', align_corners=True)
        
        if self.use_cloth_refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth)
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth
        
        return warped_cloth
    
    def get_warped_cloth(self, cloth, im_mask, pose_map):
        if self.use_cloth_warpping:
            warped_cloth = self.warpping_cloth(cloth, im_mask, pose_map)
        else:
            warped_cloth = cloth
        
        return warped_cloth
    
    def get_text_encoder_input(self, caption, category):
        if self.use_caption:
            text = [cap for
                    cap in caption]
        else:
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',
            }
            
            # batch size lenght
            text = [f'a photo of a model wearing {category_text[ctg]}' for
                    ctg in category]
        
        return text
    
    def get_tokenized_text(self, text):        
        # Tokenize text ( bsz, 77 )
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.cuda()
        
        return tokenized_text
    
    def get_encoder_hidden_states(self, text = None):
        if text is None:
            text = ""
            
        tokenized_text = self.get_tokenized_text(text)
        encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state
            
        return encoder_hidden_states
 
    def get_controlnet_hidden_blocks(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(sample=sample,
                                                                       timestep=timestep,
                                                                       encoder_hidden_states=encoder_hidden_states,
                                                                       controlnet_cond=controlnet_cond,
                                                                       conditioning_scale=self.conditioning_scale,
                                                                       return_dict=False)
        
        return down_block_res_samples, mid_block_res_sample
            
    def get_image_log(self, batch, num_sampling):
        outputs = self.inference(batch, num_sampling)

        return {"real": outputs["real"],
                "fake": outputs["fake"]}
         
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.unet.parameters(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers

    def predict(self, batch, save_dir):
        
        topil = transforms.ToPILImage()
        
        outputs = self.inference(batch)
        
        real = (outputs["real"] / 2 + 0.5).clamp(0, 1)
        fake = (outputs["fake"] / 2 + 0.5).clamp(0, 1)
        cloth = (outputs["cloth"] / 2 + 0.5).clamp(0, 1)
        w_cloth = (outputs["warped_cloth"] / 2 + 0.5).clamp(0, 1)
        
        real_pils = [topil(real_pt) for real_pt in real]
        fake_pils = [topil(fake_pt) for fake_pt in fake]
        cloth_pils = [topil(cloth_pt) for cloth_pt in cloth]
        w_cloth_pils = [topil(w_cloth_pt) for w_cloth_pt in w_cloth]
        
        for idx in range(len(real_pils)):
            fake_id = outputs['image_id'][idx].split('.')[0] + "_" + outputs['cloth_id'][idx].split('.')[0]
            
            if outputs['image_id'] == outputs['cloth_id']:
                out_dir = os.path.join(save_dir, "inference", "paired", fake_id)
            else:
                out_dir = os.path.join(save_dir, "inference", "unpaired", fake_id)
                
            os.makedirs(out_dir, exist_ok=True)
            
            real_pils[idx].save(os.path.join(out_dir, "real.png"))
            fake_pils[idx].save(os.path.join(out_dir, "fake.png"))
            cloth_pils[idx].save(os.path.join(out_dir, "cloth.png"))
            w_cloth_pils[idx].save(os.path.join(out_dir, "warped_cloth.png"))
     
class ladi_vton(Module_base):
    def __init__(self,
                 optim_target: tuple,
                 criterion_config: tuple,
                 dataset_name: str = "vitonhd", # ["vitonhd", "dresscode"]
                 in_channels: int = 31,
                 num_inference_steps: int = 50,
                 use_caption: bool = False,
                 use_cloth_warpping: bool = True,
                 use_cloth_refinement: bool = False,
                 use_img2img: bool = True,
                 model_path = None, # .../unet.ckpt
                 train_resume: bool = False
                 ):        
        super().__init__(optim_target, criterion_config, model_path, train_resume)

        self.dataset_name = dataset_name
        self.in_channels = in_channels
        self.num_inference_steps = num_inference_steps
        self.use_caption = use_caption
        self.use_cloth_warpping = use_cloth_warpping
        self.use_cloth_refinement = use_cloth_refinement
        self.use_img2img = use_img2img
        
        # Ladi-vton params
        self.emasc_int_layers = [1, 2, 3, 4, 5]
        self.num_vstar = 16
        
        # refinemnet network casuses the garment pattern to disappear.
        # I recommend not using refinemnet network.
        # If you want to use refinement network, please specify option "use_cloth_refinemnet" in the yaml file.
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module',
                                                   dataset=dataset_name)
        
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        from models.VITON.ladi_vton.models.autoencoder_kl import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        
        # Load Ladi-vton trained models from the github
        self.unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet',
                                   dataset=dataset_name)
        self.emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', 
                                    dataset=dataset_name)
        self.inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter',
                                                dataset=dataset_name)
        
        eval_models = [self.tps, self.refinement, self.vae, self.emasc, self.inversion_adapter,
                       self.text_encoder, self.vision_encoder]
        model_eval(eval_models)

        # ToDo
        # Pipeline needs to be modify for VTON
        from models.pipeline.pipeline_ladi_vton import LadiVtonPipeline as pipeline
        self.pipeline = pipeline(vae=self.vae,
                                 unet=self.unet,
                                 text_encoder=self.text_encoder,
                                 tokenizer=self.tokenizer,
                                 vision_encoder=self.vision_encoder,
                                 processor=self.processor,
                                 scheduler=self.scheduler,
                                 tps=self.tps,
                                 refinement=self.refinement,
                                 emasc=self.emasc,
                                 inversion_adapter=self.inversion_adapter)
        
    def unet_new_in_channels(self):
        if self.train_resume:
            if self.unet.config.in_channels != self.unet.conv_in.in_channels:
                raise ValueError(f"unet conv_in channels[{self.unet.conv_in.in_channels}] and unet config in_channels[{self.unet.config.in_channels}] is different.")
        else:
            # 9channels -> 31channels
            # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
            with torch.no_grad():
                # Replace the first conv layer of the unet with a new one with the correct number of input channels
                conv_new = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.unet.conv_in.out_channels,
                    kernel_size=3,
                    padding=1,
                )

            torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            conv_new.weight.data[:, :9] = self.unet.conv_in.weight.data  # Copy weights from old conv layer
            conv_new.bias.data = self.unet.conv_in.bias.data  # Copy bias from old conv layer
            
            # replace conv layer in unet
            self.unet.conv_in = conv_new  
             # update config
            self.unet.config.in_channels = self.in_channels 
            self.unet.config["in_channels"] = self.in_channels
    
    def training_resume(self):
        # This method is executed by the pl.LightningModule.
        if self.train_resume and os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            print("Unet loaded")
            self.unet.load_state_dict(torch.load(self.model_path))
         
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, latent_shape, encoder_hidden_states, zT = None, unet_input = None):
        # Not using
        pass
    
    def get_loss(self, batch):
        z0, unet_input, noise, encoder_hidden_states = self.pre_process(batch)
        unet_output = self(z0, unet_input, noise, encoder_hidden_states)
        loss = self.criterion(noise, unet_output)
        
        return {"total": loss}
    
    def pre_process(self, batch, num_sampling = None, inference = False):
        x0, c, c_w, i_m, m, p, text, _, _ = self.get_input(batch, num_sampling)
        
        down_m      = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        down_p      = nn.functional.interpolate(p, size=(p.shape[2] // 8, p.shape[3] // 8), mode="bilinear")
        z0          = self.vae.encode(x0)[0].latent_dist.sample() * self.vae.config.scaling_factor
        latent_c_w  = self.vae.encode(c_w)[0].latent_dist.sample() * self.vae.config.scaling_factor
        latent_i_m  = self.vae.encode(i_m)[0].latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input  = torch.cat([down_m, down_p, latent_c_w, latent_i_m], 1)
        noise       = torch.randn(z0.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.get_encoder_hidden_states(bsz=c.size(0), image=c, text=text)
        
        return z0, unet_input, noise, encoder_hidden_states
    
    def forward(self, z0, unet_input, noise, encoder_hidden_states):
        timestep = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zt = self.forward_diffusion_process(z0, noise, timestep)
        sample = torch.cat([zt, unet_input], 1)
        
        rec_sample = self.unet(sample=sample, 
                               timestep=timestep, 
                               encoder_hidden_states=encoder_hidden_states).sample
        
        return rec_sample
    
    def inference(self, batch, num_sampling=None):
        image, cloth, warped_cloth_image, _, mask_image, posemap, prompt, im_name, c_name = self.get_input(batch, num_sampling)
        
        self.pipeline.to("cuda")
        x0_pred = self.pipeline(prompt=prompt,
                                image=image,
                                mask_image=mask_image,
                                cloth_image=cloth,
                                warped_cloth_image=warped_cloth_image,
                                posemap_image=posemap,
                                height = 512,
                                width = 384,
                                num_inference_steps=self.num_inference_steps,
                                output_type="pt",
                                use_cloth_warpping=self.use_cloth_warpping,
                                use_cloth_refinemnet=self.use_cloth_refinement,
                                emasc_int_layers=self.emasc_int_layers,
                                num_vstar=self.num_vstar
                                ).images
        # re normalize
        ## because pipeline has denormalize
        x0_pred = 2.0 * x0_pred - 1.0
        
        return {"real": image, 
                "fake": x0_pred,
                "cloth": cloth,
                "warped_cloth": warped_cloth_image,
                "image_id": im_name,
                "cloth_id": c_name}
    
    def get_input(self, batch, num_sampling = None):
        i = batch["image"]
        c = batch["cloth"]
        i_m = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        p = batch["pose_map"]
        c_w = batch.get("warped_cloth") if batch.get("warped_cloth") is not None else self.get_warped_cloth(c, i_m, p)
        text = self.get_text_encoder_input(batch["captions"], batch["category"])
        
        im_name = batch["im_name"]
        c_name = batch["c_name"]
        
        if num_sampling is not None:
            i = i[:num_sampling]
            c = c[:num_sampling]
            i_m = i_m[:num_sampling]
            m = m[:num_sampling]
            p = p[:num_sampling]
            c_w = c_w[:num_sampling]
            text = text[:num_sampling]
            
            im_name = im_name[:num_sampling]
            c_name = c_name[:num_sampling]
            
        return i, c, c_w, i_m, m, p, text, im_name, c_name
        
    def warpping_cloth(self, cloth, im_mask, pose_map):
        # TPS parameters prediction
        # For sake of performance, the TPS parameters are predicted on a low resolution image
        low_cloth = transforms.functional.resize(cloth, (256, 192),
                                                 transforms.InterpolationMode.BILINEAR,
                                                 antialias=True)
        low_im_mask = transforms.functional.resize(im_mask, (256, 192),
                                                   transforms.InterpolationMode.BILINEAR,
                                                   antialias=True)
        low_pose_map = transforms.functional.resize(pose_map, (256, 192),
                                                    transforms.InterpolationMode.BILINEAR,
                                                    antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, _, _, _, _, _, _, _ = self.tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(cloth.size(2), cloth.size(3)), 
                               interpolation=transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)

        warped_cloth = nn.functional.grid_sample(cloth, highres_grid, padding_mode='border', align_corners=True)
        
        if self.use_cloth_refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth)
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth
        
        return warped_cloth
    
    def get_warped_cloth(self, cloth, im_mask, pose_map):
        if self.use_cloth_warpping:
            warped_cloth = self.warpping_cloth(cloth, im_mask, pose_map)
        else:
            warped_cloth = cloth
        
        return warped_cloth
    
    def get_text_encoder_input(self, caption, category):
        if self.use_caption:
            text = [cap for
                    cap in caption]
        else:
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',
            }
            
            # batch size lenght
            text = [f'a photo of a model wearing {category_text[ctg]}' for
                    ctg in category]
        
        return text
    
    def get_word_embedding(self, image):
        # Get the visual features of the in-shop cloths
        # (bsz, 3, 224, 224)
        input_image = transforms.functional.resize((image + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
        # (bsz, 3, 224, 224)
        processed_images = self.processor(images=input_image, return_tensors="pt", do_rescale=False)
        # (bsz, 257, 1280)
        clip_image_features = self.vision_encoder(processed_images.pixel_values.cuda()).last_hidden_state
        # Compute the predicted PTEs
        # (bsz, 16384)
        word_embeddings = self.inversion_adapter(clip_image_features)
        # (bsz, 16, 1024)
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], self.num_vstar, -1))
        
        return word_embeddings
    
    def get_tokenized_text(self, text):        
        # Tokenize text ( bsz, 77 )
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.cuda()
        
        return tokenized_text
    
    def get_encoder_hidden_states(self, bsz, image = None, text = None):
        if text is None:
            text = [f' {"$" * self.num_vstar}' for
                    _ in range(bsz)]
        if image is None:
            tokenized_text = self.get_tokenized_text(text)
            encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state
        else:
            word_embeddings = self.get_word_embedding(image)
            tokenized_text = self.get_tokenized_text(text)
            
            from models.Diffusion.ladi_vton.utils.encode_text_word_embedding import encode_text_word_embedding
            # Encode the text using the PTEs extracted from the in-shop cloths ( bsz, 77, 1024 )
            encoder_hidden_states = encode_text_word_embedding(self.text_encoder, tokenized_text,
                                                               word_embeddings, self.num_vstar).last_hidden_state
            
        return encoder_hidden_states
     
    def get_image_log(self, batch, num_sampling):
        outputs = self.inference(batch, num_sampling)

        return {"real": outputs["real"],
                "fake": outputs["fake"]}
                
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.unet.parameters(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers

    def predict(self, batch, save_dir):
        
        topil = transforms.ToPILImage()
        
        outputs = self.inference(batch)
        
        real = (outputs["real"] / 2 + 0.5).clamp(0, 1)
        fake = (outputs["fake"] / 2 + 0.5).clamp(0, 1)
        cloth = (outputs["cloth"] / 2 + 0.5).clamp(0, 1)
        w_cloth = (outputs["warped_cloth"] / 2 + 0.5).clamp(0, 1)
        
        real_pils = [topil(real_pt) for real_pt in real]
        fake_pils = [topil(fake_pt) for fake_pt in fake]
        cloth_pils = [topil(cloth_pt) for cloth_pt in cloth]
        w_cloth_pils = [topil(w_cloth_pt) for w_cloth_pt in w_cloth]
        
        for idx in range(len(real_pils)):
            fake_id = outputs['image_id'][idx].split('.')[0] + "_" + outputs['cloth_id'][idx].split('.')[0]
            
            if outputs['image_id'] == outputs['cloth_id']:
                out_dir = os.path.join(save_dir, "inference", "paired", fake_id)
            else:
                out_dir = os.path.join(save_dir, "inference", "unpaired", fake_id)
                
            os.makedirs(out_dir, exist_ok=True)
            
            real_pils[idx].save(os.path.join(out_dir, "real.png"))
            fake_pils[idx].save(os.path.join(out_dir, "fake.png"))
            cloth_pils[idx].save(os.path.join(out_dir, "cloth.png"))
            w_cloth_pils[idx].save(os.path.join(out_dir, "warped_cloth.png"))
            
# ToDo
class ladi_vton_with_controlnet(Module_base):
    def __init__(self,
                 optim_target: tuple,
                 criterion_config: tuple,
                 dataset_name: str = "vitonhd", # ["vitonhd", "dresscode"]
                 in_channels: int = 31,
                 num_inference_steps: int = 50,
                 use_caption: bool = False,
                 use_cloth_warpping: bool = True,
                 use_cloth_refinement: bool = False,
                 use_img2img: bool = True,
                 conditioning_scale = 1.0,
                 model_path = None, # .../unet.ckpt
                 train_resume: bool = False
                 ):        
        super().__init__(optim_target, criterion_config, model_path, train_resume)

        self.dataset_name = dataset_name
        self.in_channels = in_channels
        self.num_inference_steps = num_inference_steps
        self.use_caption = use_caption
        self.use_cloth_warpping = use_cloth_warpping
        self.use_cloth_refinement = use_cloth_refinement
        self.use_img2img = use_img2img
        self.conditioning_scale = conditioning_scale
        
        # Ladi-vton params
        self.emasc_int_layers = [1, 2, 3, 4, 5]
        self.num_vstar = 16
        
        # refinemnet network casuses the garment pattern to disappear.
        # I recommend not using refinemnet network.
        # If you want to use refinement network, please specify option "use_cloth_refinemnet" in the yaml file.
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module',
                                                   dataset=dataset_name)
        
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        from models.VITON.ladi_vton.models.autoencoder_kl import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        
        # Load Ladi-vton trained models from the github
        self.unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet',
                                   dataset=dataset_name)
        self.controlnet = ControlNetModel.from_unet(unet=self.unet,
                                                    conditioning_embedding_out_channels=[16, 32, 96, 256])
        self.emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', 
                                    dataset=dataset_name)
        self.inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter',
                                                dataset=dataset_name)
        
        eval_models = [self.tps, self.refinement, self.vae, self.emasc, self.inversion_adapter, self.unet,
                       self.text_encoder, self.vision_encoder]
        model_eval(eval_models)

        # ToDo
        # Pipeline needs to be modify for VTON
        from models.pipeline.pipeline_ladi_vton_with_controlnet import LadiVtonWithControlnetPipeline as pipeline
        self.pipeline = pipeline(vae=self.vae,
                                 unet=self.unet,
                                 controlnet=self.controlnet,
                                 text_encoder=self.text_encoder,
                                 tokenizer=self.tokenizer,
                                 vision_encoder=self.vision_encoder,
                                 processor=self.processor,
                                 scheduler=self.scheduler,
                                 tps=self.tps,
                                 refinement=self.refinement,
                                 emasc=self.emasc,
                                 inversion_adapter=self.inversion_adapter)
        
    def unet_new_in_channels(self):
        if self.train_resume:
            if self.unet.config.in_channels != self.unet.conv_in.in_channels:
                raise ValueError(f"unet conv_in channels[{self.unet.conv_in.in_channels}] and unet config in_channels[{self.unet.config.in_channels}] is different.")
        else:
            # 9channels -> 31channels
            # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
            with torch.no_grad():
                # Replace the first conv layer of the unet with a new one with the correct number of input channels
                conv_new = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.unet.conv_in.out_channels,
                    kernel_size=3,
                    padding=1,
                )

            torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            conv_new.weight.data[:, :9] = self.unet.conv_in.weight.data  # Copy weights from old conv layer
            conv_new.bias.data = self.unet.conv_in.bias.data  # Copy bias from old conv layer
            
            # replace conv layer in unet
            self.unet.conv_in = conv_new  
             # update config
            self.unet.config.in_channels = self.in_channels 
            self.unet.config["in_channels"] = self.in_channels
    
    def training_resume(self):
        # This method is executed by the pl.LightningModule.
        if self.train_resume and os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            print("Unet loaded")
            self.unet.load_state_dict(torch.load(self.model_path))
         
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, latent_shape, encoder_hidden_states, zT = None, unet_input = None):
        # Not using
        pass
    
    def get_loss(self, batch):
        z0, unet_input, noise, encoder_hidden_states, controlnet_cond = self.pre_process(batch)
        unet_output = self(z0, unet_input, noise, encoder_hidden_states, controlnet_cond)
        loss = self.criterion(noise, unet_output)
        
        return {"total": loss}
    
    def pre_process(self, batch, num_sampling = None, inference = False):
        x0, c, c_w, i_m, m, p, text, _, _ = self.get_input(batch, num_sampling)
        
        down_m      = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        down_p      = nn.functional.interpolate(p, size=(p.shape[2] // 8, p.shape[3] // 8), mode="bilinear")
        z0          = self.vae.encode(x0)[0].latent_dist.sample() * self.vae.config.scaling_factor
        latent_c_w  = self.vae.encode(c_w)[0].latent_dist.sample() * self.vae.config.scaling_factor
        latent_i_m  = self.vae.encode(i_m)[0].latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input  = torch.cat([down_m, down_p, latent_c_w, latent_i_m], 1)
        noise       = torch.randn(z0.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.get_encoder_hidden_states(bsz=c.size(0), image=c, text=text)
        controlnet_cond = c_w
        
        return z0, unet_input, noise, encoder_hidden_states, controlnet_cond
    
    def forward(self, z0, unet_input, noise, encoder_hidden_states, controlnet_cond):
        timestep = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0), ), dtype=torch.long, device=z0.device)
        zt = self.forward_diffusion_process(z0, noise, timestep)
        sample = torch.cat([zt, unet_input], 1)
        down_block_res_samples, mid_block_res_sample = self.get_controlnet_hidden_blocks(sample, timestep, encoder_hidden_states, controlnet_cond)
        
        rec_sample = self.unet(sample=sample, 
                               timestep=timestep, 
                               encoder_hidden_states=encoder_hidden_states,
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample).sample
        
        return rec_sample
    
    def inference(self, batch, num_sampling=None):
        image, cloth, warped_cloth_image, _, mask_image, posemap, prompt, im_name, c_name = self.get_input(batch, num_sampling)
        
        self.pipeline.to("cuda")
        x0_pred = self.pipeline(prompt=prompt,
                                image=image,
                                mask_image=mask_image,
                                cloth_image=cloth,
                                warped_cloth_image=warped_cloth_image,
                                posemap_image=posemap,
                                height = 512,
                                width = 384,
                                num_inference_steps=self.num_inference_steps,
                                output_type="pt",
                                use_cloth_warpping=self.use_cloth_warpping,
                                use_cloth_refinemnet=self.use_cloth_refinement,
                                emasc_int_layers=self.emasc_int_layers,
                                num_vstar=self.num_vstar
                                ).images
        # re normalize
        ## because pipeline has denormalize
        x0_pred = 2.0 * x0_pred - 1.0
        
        return {"real": image, 
                "fake": x0_pred,
                "cloth": cloth,
                "warped_cloth": warped_cloth_image,
                "image_id": im_name,
                "cloth_id": c_name}
    
    def get_input(self, batch, num_sampling = None):
        i = batch["image"]
        c = batch["cloth"]
        i_m = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        p = batch["pose_map"]
        c_w = batch.get("warped_cloth") if batch.get("warped_cloth") is not None else self.get_warped_cloth(c, i_m, p)
        text = self.get_text_encoder_input(batch["captions"], batch["category"])
        
        im_name = batch["im_name"]
        c_name = batch["c_name"]
        
        if num_sampling is not None:
            i = i[:num_sampling]
            c = c[:num_sampling]
            i_m = i_m[:num_sampling]
            m = m[:num_sampling]
            p = p[:num_sampling]
            c_w = c_w[:num_sampling]
            text = text[:num_sampling]
            
            im_name = im_name[:num_sampling]
            c_name = c_name[:num_sampling]
            
        return i, c, c_w, i_m, m, p, text, im_name, c_name
        
    def warpping_cloth(self, cloth, im_mask, pose_map):
        # TPS parameters prediction
        # For sake of performance, the TPS parameters are predicted on a low resolution image
        low_cloth = transforms.functional.resize(cloth, (256, 192),
                                                 transforms.InterpolationMode.BILINEAR,
                                                 antialias=True)
        low_im_mask = transforms.functional.resize(im_mask, (256, 192),
                                                   transforms.InterpolationMode.BILINEAR,
                                                   antialias=True)
        low_pose_map = transforms.functional.resize(pose_map, (256, 192),
                                                    transforms.InterpolationMode.BILINEAR,
                                                    antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, _, _, _, _, _, _, _ = self.tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(cloth.size(2), cloth.size(3)), 
                               interpolation=transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)

        warped_cloth = nn.functional.grid_sample(cloth, highres_grid, padding_mode='border', align_corners=True)
        
        if self.use_cloth_refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth)
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth
        
        return warped_cloth
    
    def get_warped_cloth(self, cloth, im_mask, pose_map):
        if self.use_cloth_warpping:
            warped_cloth = self.warpping_cloth(cloth, im_mask, pose_map)
        else:
            warped_cloth = cloth
        
        return warped_cloth
    
    def get_text_encoder_input(self, caption, category):
        if self.use_caption:
            text = [cap for
                    cap in caption]
        else:
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',
            }
            
            # batch size lenght
            text = [f'a photo of a model wearing {category_text[ctg]}' for
                    ctg in category]
        
        return text
    
    def get_word_embedding(self, image):
        # Get the visual features of the in-shop cloths
        # (bsz, 3, 224, 224)
        input_image = transforms.functional.resize((image + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
        # (bsz, 3, 224, 224)
        processed_images = self.processor(images=input_image, return_tensors="pt", do_rescale=False)
        # (bsz, 257, 1280)
        clip_image_features = self.vision_encoder(processed_images.pixel_values.cuda()).last_hidden_state
        # Compute the predicted PTEs
        # (bsz, 16384)
        word_embeddings = self.inversion_adapter(clip_image_features)
        # (bsz, 16, 1024)
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], self.num_vstar, -1))
        
        return word_embeddings
    
    def get_tokenized_text(self, text):        
        # Tokenize text ( bsz, 77 )
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.cuda()
        
        return tokenized_text
    
    def get_encoder_hidden_states(self, bsz, image = None, text = None):
        if text is None:
            text = [f' {"$" * self.num_vstar}' for
                    _ in range(bsz)]
        if image is None:
            tokenized_text = self.get_tokenized_text(text)
            encoder_hidden_states = self.text_encoder(tokenized_text).last_hidden_state
        else:
            word_embeddings = self.get_word_embedding(image)
            tokenized_text = self.get_tokenized_text(text)
            
            from models.Diffusion.ladi_vton.utils.encode_text_word_embedding import encode_text_word_embedding
            # Encode the text using the PTEs extracted from the in-shop cloths ( bsz, 77, 1024 )
            encoder_hidden_states = encode_text_word_embedding(self.text_encoder, tokenized_text,
                                                               word_embeddings, self.num_vstar).last_hidden_state
            
        return encoder_hidden_states
     
    def get_controlnet_hidden_blocks(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(sample=sample,
                                                                       timestep=timestep,
                                                                       encoder_hidden_states=encoder_hidden_states,
                                                                       controlnet_cond=controlnet_cond,
                                                                       conditioning_scale=self.conditioning_scale,
                                                                       return_dict=False)
        
        return down_block_res_samples, mid_block_res_sample
        
    def get_image_log(self, batch, num_sampling):
        outputs = self.inference(batch, num_sampling)

        return {"real": outputs["real"],
                "fake": outputs["fake"]}
                
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.unet.parameters(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers

    def predict(self, batch, save_dir):
        
        topil = transforms.ToPILImage()
        
        outputs = self.inference(batch)
        
        real = (outputs["real"] / 2 + 0.5).clamp(0, 1)
        fake = (outputs["fake"] / 2 + 0.5).clamp(0, 1)
        cloth = (outputs["cloth"] / 2 + 0.5).clamp(0, 1)
        w_cloth = (outputs["warped_cloth"] / 2 + 0.5).clamp(0, 1)
        
        real_pils = [topil(real_pt) for real_pt in real]
        fake_pils = [topil(fake_pt) for fake_pt in fake]
        cloth_pils = [topil(cloth_pt) for cloth_pt in cloth]
        w_cloth_pils = [topil(w_cloth_pt) for w_cloth_pt in w_cloth]
        
        for idx in range(len(real_pils)):
            fake_id = outputs['image_id'][idx].split('.')[0] + "_" + outputs['cloth_id'][idx].split('.')[0]
            
            if outputs['image_id'] == outputs['cloth_id']:
                out_dir = os.path.join(save_dir, "inference", "paired", fake_id)
            else:
                out_dir = os.path.join(save_dir, "inference", "unpaired", fake_id)
                
            os.makedirs(out_dir, exist_ok=True)
            
            real_pils[idx].save(os.path.join(out_dir, "real.png"))
            fake_pils[idx].save(os.path.join(out_dir, "fake.png"))
            cloth_pils[idx].save(os.path.join(out_dir, "cloth.png"))
            w_cloth_pils[idx].save(os.path.join(out_dir, "warped_cloth.png"))