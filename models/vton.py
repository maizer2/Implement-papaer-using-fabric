import importlib, os
from typing import Callable, Union, List, Tuple, Dict

import lightning.pytorch as pl

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import make_grid
from torchvision import transforms

from diffusers import UNet2DConditionModel, StableDiffusionInpaintPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from models.VITON.ladi_vton.models.inversion_adapter import InversionAdapter
from models.VITON.ladi_vton.utils.encode_text_word_embedding import encode_text_word_embedding

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
    
class ladi_vton(Module_base):
    def __init__(self,
                 optim_target: tuple,
                 criterion_config: tuple,
                 stage: str, # ["emasc", "inversion_adapter", "tryon"]
                 dataset_name: str, # ["vitonhd", "dresscode"]
                 scheduler_config: dict = None,
                 emasc_config: dict = None,
                 inversion_adapter_config: dict = None,
                 in_channels: int = 31,
                 num_inference_steps = 50,
                 img2img = False,
                 cloth_warpping = False,
                 cloth_refinement = False,
                 use_emasc = False,
                 model_path = None):
        super().__init__(model_path)
        self.optimizer = get_obj_from_str(optim_target)
        self.criterion = instantiate_from_config(criterion_config)
        
        self.stage = stage
        self.dataset_name = dataset_name
        self.in_channels = in_channels
        self.num_inference_steps = num_inference_steps
        self.img2img = img2img
        self.cloth_warpping = cloth_warpping
        self.cloth_refinement = cloth_refinement
        self.use_emasc = use_emasc
        self.model_path = model_path
        
        
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module', 
                                                   dataset=self.dataset_name)
        
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        
        from models.VITON.ladi_vton.models.autoencoder_kl import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")

        model_eval([self.tps, self.refinement, self.vae, self.text_encoder])
        self.unet_init(in_channels)
        
        if stage == "emasc":
            self.emasc = instantiate_from_config(emasc_config)
            model_eval([self.emasc])
            
            if os.path.exists(model_path):
                self.emasc.load_state_dict(torch.load(model_path))
        elif stage == "inversion_adapter":
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.inversion_adapter = InversionAdapter(**inversion_adapter_config,
                                                     clip_config=self.vision_encoder.config)
            self.scheduler = instantiate_from_config(scheduler_config)
            model_eval([self.vision_encoder, self.inversion_adapter])
            
            if os.path.exists(model_path):
                self.inversion_adapter.load_state_dict(torch.load(model_path))
        else:
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.emasc = instantiate_from_config(emasc_config)
            self.inversion_adapter = InversionAdapter(**inversion_adapter_config,
                                                     clip_config=self.vision_encoder.config,
                                                     output_dim=self.text_encoder.config.hidden_size * inversion_adapter_config["num_vstar"])
            self.scheduler = instantiate_from_config(scheduler_config)
            model_eval([self.vision_encoder, self.emasc, self.inversion_adapter])
            
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
            
    def train_tps(self, batch):
        loss = None
        
        return [("total", loss)]
    
    def train_emasc(self, batch, batch_idx, epoch):
        image = batch["image"]
        fake = self.vae(image).sample
        
        # latents = self.vae.encode(image)[0].latent_dist.sample() * self.vae.config.scaling_factor
        
        for idx in range(image.size(0)):
            grid = make_grid(torch.cat([image[idx].unsqueeze(0), fake[idx].unsqueeze(0)], 0), normalize=True)
            transforms.ToPILImage()(grid).save(f"test/{epoch}_{batch_idx}.png")

        return torch.tensor(1.0, requires_grad=True).cuda()
    
    def train_inversion_adapter(self, batch):
        loss = None
        
        return [("total", loss)]
    
    def train_tryon(self, batch):
        latent, unet_input, noise, encoder_hidden_states, _ = self.get_tryon_input(batch)
        # denoising을 전체적으로? 아님 z만? -> z만
        unet_output = self(latent, unet_input, noise, encoder_hidden_states)
        
        loss = self.criterion(noise, unet_output)
        
        return [("total", loss)]
    
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, zT = None, shape = None, unet_input = None, prompt_embeds = None) -> torch.from_numpy:
        if zT is None:
            zT = torch.randn(shape, dtype=torch.float32).cuda()
                
        if unet_input is None:
            unet_shape = (shape[0], self.in_channels - zT.size(1), shape[2], shape[3])
            unet_input = torch.randn(unet_shape, dtype=zT.dtype).cuda()
            
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(["" * shape[0]])
            
        pred_z0 = zT
        self.scheduler.set_timesteps(self.num_inference_steps, zT.device)
        for t in self.scheduler.timesteps:
            
            input = torch.cat([pred_z0, unet_input], 1)
            # 1. predict noise model_output
            model_output = self.unet(input, t, prompt_embeds).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_z0 = self.scheduler.step(model_output, t, pred_z0).prev_sample
        
        return pred_z0
    
    def forward(self, latent, unet_input, noise, prompt_embeds):
        t = torch.randint(0, len(self.scheduler.timesteps), (latent.size(0), ), dtype=torch.long, device=latent.device)
        zt = self.forward_diffusion_process(latent, noise, t)
        zt = torch.cat([zt, unet_input], 1)
        
        rec_sample = self.unet(zt, t, prompt_embeds).sample
        
        return rec_sample
    
    def get_tryon_input(self, batch, num_sampling = None):
        I, C, C_W, I_M, m, p = self.get_input(batch, num_sampling)
        
        m = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        p = nn.functional.interpolate(p, size=(p.shape[2] // 8, p.shape[3] // 8), mode="bilinear")
        latent = self.vae.encode(I)[0].latent_dist.sample() * self.vae.config.scaling_factor
        C_W = self.vae.encode(C_W)[0].latent_dist.sample() * self.vae.config.scaling_factor
        I_M, I_M_intermediate_features = self.vae.encode(I_M)
        I_M = I_M.latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input = torch.cat([m, p, C_W, I_M], 1)
        noise = torch.randn(latent.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.get_encoder_hidden_states(C, batch["category"])
        
        I_M_intermediate_features = self.get_intermediate_features(I_M_intermediate_features)
        
        return latent, unet_input, noise, encoder_hidden_states, I_M_intermediate_features
    
    def get_input(self, batch, num_sampling = None):
        I = batch["image"]
        C = batch["cloth"]
        I_M = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        p = batch["pose_map"]
        C_W = self.get_warped_cloth(batch)
        
        if num_sampling is not None:
            I = I[:num_sampling]
            C = C[:num_sampling]
            I_M = I_M[:num_sampling]
            m = m[:num_sampling]
            p = p[:num_sampling]
            C_W = C_W[:num_sampling]
        
                        
        return I, C, C_W, I_M, m, p
    
    def get_loss(self, batch, batch_idx, epoch) -> List[Tuple[str, int]]:
        if self.stage == "emasc":
            losses = self.train_emasc(batch, batch_idx, epoch)
        elif self.stage == "inversion_adapter":
            losses = self.train_inversion_adapter(batch)
        else:
            losses = self.train_tryon(batch)
            
        return losses
    
    def tryon_inference(self, batch, num_sampling):
        latent, unet_input, noise, encoder_hidden_states, I_M_intermediate_features = self.get_tryon_input(batch, num_sampling)
        
        self.unet.eval()
        
        with torch.no_grad():
            if self.img2img:
                zT = self.forward_diffusion_process(latent, noise)
            else:
                zT = None
            
            pred_z0 = self.reverse_diffusion_process(zT, latent.shape, unet_input, encoder_hidden_states)
            if self.use_emasc:
                pred_x0 = self.vae.decode(pred_z0, I_M_intermediate_features, self.emasc.int_layers).sample
            else:
                pred_x0 = self.vae.decode(pred_z0).sample
                
        self.unet.train()
        
        return pred_x0
    
    def inference(self, batch, num_sampling) -> torch.Tensor:
        if self.stage == "tryon":
            pred = self.tryon_inference(batch, num_sampling)
            
        return pred
    
    def get_image_log(self, batch, num_sampling) -> Dict[str, torch.Tensor]:
        I, C, C_W, I_M, m, _ = self.get_input(batch, num_sampling)
        pred = self.inference(batch, num_sampling)
        
        return {"image": I,
                "cloth": C,
                "warped_cloth": C_W,
                "image_mask": I_M,
                "inpaint_mask": m,
                "skeleton": batch["skeleton"],
                "result": pred}
    
    def save_model(self):
        if self.stage == "tryon":
            torch.save(self.unet.state_dict(), self.model_path)
        
    def warping_cloth(self, batch):
        cloth = batch["cloth"]
        im_mask = batch["im_mask"]
        pose_map = batch["pose_map"]
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
        
        if self.cloth_refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth)
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth
        
        return warped_cloth
    
    def get_warped_cloth(self, batch):
        warped_cloth = batch.get("warped_cloth")
        
        if warped_cloth is None:
            if self.cloth_warpping:
                warped_cloth = self.warping_cloth(batch)
        else:
            warped_cloth = batch.get("cloth")
        
        return warped_cloth
    
    def get_intermediate_features(self, intermediate_features):
        intermediate_features = [intermediate_features[i] for i in self.emasc.int_layers]
        processed_intermediate_features = self.emasc(intermediate_features)
        
        return processed_intermediate_features
    
    def get_word_embedding(self, cloth):
        # Get the visual features of the in-shop cloths
        # (bsz, 3, 224, 224)
        input_image = transforms.functional.resize((cloth + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
        # (bsz, 3, 224, 224)
        processed_images = self.processor(images=input_image, return_tensors="pt", do_rescale=False)
        # (bsz, 257, 1280)
        clip_cloth_features = self.vision_encoder(processed_images.pixel_values.cuda()).last_hidden_state
        # Compute the predicted PTEs
        # (bsz, 16384)
        word_embeddings = self.inversion_adapter(clip_cloth_features)
        # (bsz, 16, 1024)
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], self.inversion_adapter.num_vstar, -1))
        
        return word_embeddings
    
    def get_tokenized_text(self, category):
        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',
        }
        
        # batch size lenght
        text = [f'a photo of a model wearing {category_text[ctg]} {" $ " * self.inversion_adapter.num_vstar}' for
                ctg in category]
        
        # Tokenize text ( bsz, 77 )
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids.cuda()
        
        return tokenized_text
    
    def get_encoder_hidden_states(self, cloth, category):
        word_embeddings = self.get_word_embedding(cloth)
        tokenized_text = self.get_tokenized_text(category)
        
        # Encode the text using the PTEs extracted from the in-shop cloths ( bsz, 77, 1024 )
        encoder_hidden_states = encode_text_word_embedding(self.text_encoder, tokenized_text,
                                                           word_embeddings, self.inversion_adapter.num_vstar).last_hidden_state
        
        return encoder_hidden_states
     
class custom_vton(Module_base):
    def __init__(self, 
                 optim_target: str,
                 criterion_config: tuple,
                 scheduler_config: tuple,
                 dataset_name: str, # ["vitonhd", "dresscode"]
                 in_channels: int = 31,
                 num_inference_steps: int = 50,
                 cloth_warpping: bool = False,
                 cloth_refinement: bool = False,
                 img2img: bool = False,
                 model_path: str = None
                 ):
        super().__init__(model_path)
        self.optimizer = get_obj_from_str(optim_target)
        self.criterion = instantiate_from_config(criterion_config)
        
        self.in_channels = in_channels
        self.num_inference_steps = num_inference_steps
        self.cloth_warpping = cloth_warpping
        self.cloth_refinement = cloth_refinement
        self.img2img = img2img
        
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module', 
                                                   dataset=dataset_name)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
        
        from diffusers.models.autoencoder_kl import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
        self.scheduler = instantiate_from_config(scheduler_config)

        self.unet_init(in_channels)
        
        if os.path.exists(model_path):
            self.unet.load_state_dict(torch.load(model_path))

        model_eval([self.tps, self.refinement, self.text_encoder, self.vae])
        
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
            
    def forward_diffusion_process(self, z0, noise = None, t = None) -> torch.from_numpy:
        if noise is None:
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device)
        
        if t is None:
            t = torch.full((z0.size(0), ), self.scheduler.timesteps[0], dtype=torch.long, device=z0.device)
            
        zT = self.scheduler.add_noise(z0, noise, t)
    
        return zT
    
    def reverse_diffusion_process(self, zT = None, shape = None, unet_input = None, prompt_embeds = None) -> torch.from_numpy:
        if zT is None:
            zT = torch.randn(shape, dtype=torch.float32).cuda()
                
        if unet_input is None:
            unet_shape = (shape[0], self.in_channels - zT.size(1), shape[2], shape[3])
            unet_input = torch.randn(unet_shape, dtype=zT.dtype).cuda()
            
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(["" * shape[0]])
            
        pred_z0 = zT
        self.scheduler.set_timesteps(self.num_inference_steps, zT.device)
        for t in self.scheduler.timesteps:
            
            input = torch.cat([pred_z0, unet_input], 1)
            # 1. predict noise model_output
            model_output = self.unet(input, t, prompt_embeds).sample

            # 2. compute previous image: x_t -> x_t-1
            pred_z0 = self.scheduler.step(model_output, t, pred_z0).prev_sample
        
        return pred_z0
    
    def forward(self, latent, unet_input, noise, encoder_hidden_states):
        t = torch.randint(0, len(self.scheduler.timesteps), (latent.size(0), ), dtype=torch.long, device=latent.device)
        zt = self.forward_diffusion_process(latent, noise, t)
        zt = torch.cat([zt, unet_input], 1)
        
        rec_sample = self.unet(zt, t, encoder_hidden_states).sample
        
        return rec_sample
    
    def warping_cloth(self, batch):
        cloth = batch["cloth"]
        im_mask = batch["im_mask"]
        pose_map = batch["pose_map"]
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
        
        if self.cloth_refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = self.refinement(warped_cloth)
            warped_cloth = warped_cloth.clamp(-1, 1)
            warped_cloth = warped_cloth
        
        return warped_cloth
    
    def get_warped_cloth(self, batch):
        warped_cloth = batch.get("warped_cloth")
        
        if warped_cloth is None:
            if self.cloth_warpping:
                warped_cloth = self.warping_cloth(batch)
            else:
                warped_cloth = batch.get("cloth")
        
        return warped_cloth
    
    def get_input(self, batch, num_sampling = None):
        I = batch["image"]
        C = batch["cloth"]
        I_M = batch["im_mask"]
        m = batch["inpaint_mask"].to(torch.float32)
        p = batch["pose_map"]
        C_W = self.get_warped_cloth(batch)
        
        if num_sampling is not None:
            I = I[:num_sampling]
            C = C[:num_sampling]
            I_M = I_M[:num_sampling]
            m = m[:num_sampling]
            p = p[:num_sampling]
            C_W = C_W[:num_sampling]
        
        m = nn.functional.interpolate(m, size=(m.shape[2] // 8, m.shape[3] // 8), mode="bilinear")
        p = nn.functional.interpolate(p, size=(p.shape[2] // 8, p.shape[3] // 8), mode="bilinear")
        latent = self.vae.encode(I).latent_dist.sample() * self.vae.config.scaling_factor
        C_W = self.vae.encode(C_W).latent_dist.sample() * self.vae.config.scaling_factor
        I_M = self.vae.encode(I_M).latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input = torch.cat([m, p, C_W, I_M], 1)
        noise = torch.randn(latent.shape, dtype=torch.float32).cuda()
        encoder_hidden_states = self.encode_prompt(batch["captions"])
        
        return latent, unet_input, noise, encoder_hidden_states
    
    def get_loss(self, batch):
        latent, unet_input, noise, encoder_hidden_states = self.get_input(batch)
        # denoising을 전체적으로? 아님 z만? -> z만
        unet_output = self(latent, unet_input, noise, encoder_hidden_states)
        
        loss = self.criterion(noise, unet_output)
        
        return {"total": loss}
        
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
 
    def get_image_log(self, batch, num_sampling):
        z0, unet_input, noise, encoder_hidden_states = self.get_input(batch, num_sampling)
        
        if self.img2img:
            zT = self.forward_diffusion_process(z0, noise)
        else:
            zT = None
            
        z0_pred = self.reverse_diffusion_process(zT=zT, 
                                                 shape=z0.shape,
                                                 unet_input=unet_input,
                                                 prompt_embeds=encoder_hidden_states)
        
        x0_pred = self.vae.decode(z0_pred).sample

        return {"real": batch["image"],
                "fake": x0_pred}
        
    def save_model(self):
        torch.save(self.unet.state_dict(), self.model_path)
        
    def configure_optimizers(self, lr):
        optim = self.optimizer(self.unet.parameters(), lr)
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler = LambdaLR(optim, lambda2)
        
        optimizers = [optim]
        schedulers = [scheduler]
        
        return optimizers, schedulers
    