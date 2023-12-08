import importlib, os
from einops import rearrange
from typing import Optional, Union, List, Tuple, Dict, Any

import lightning.pytorch as pl

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import make_grid
from torchvision import transforms

from models.Diffusion.Frido.taming.modules.diffusionmodules.model import Decoder
from models.Diffusion.Frido.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from run import instantiate_from_config

class msvqgan(nn.Module):
    def __init__(self, 
                 optim_name: str,
                 encoder_config: tuple,
                 decoder_config: tuple,
                 loss_config: tuple,
                 embed_dim: List[int, int] = [4, 4],
                 n_embed: List[int, int] = [8192, 8192],
                 fusion: str = 'concat',
                 use_aux_loss: bool = False,
                 unsample_type: str = 'nearest',
                 remap = None,
                 sane_index_shape: bool = False,
                 quant_beta: float = 0.25,
                 init_norm: bool = False,
                 legacy=True,
                 colorize_nlabels=None,
                 model_path: str = None
                 ):
        super().__init__()
        self.optimizer = getattr(importlib.import_module("torch.optim"), optim_name)
        self.criterion = instantiate_from_config(loss_config)
        
        self.fusion = fusion
        self.embed_dim = embed_dim
        self.use_aux_loss = use_aux_loss
        self.unsample_type = unsample_type
        self.model_path = model_path
        
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.ms_quantize = nn.ModuleList()
        self.ms_quant_conv = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.shared_decoder = nn.ModuleList()
        self.shared_post_quant_conv = nn.ModuleList()
        
        assert len(n_embed) == encoder_config['multiscale'], 'multiscale mode. dim of n_embed is incorrect.'
        assert len(n_embed) == len(embed_dim), 'multiscale mode. dim of n_embed is incorrect.'

        self.res_list = []
        for i in range(self.encoder.multiscale):
            self.res_list.append(self.encoder.resolution / 2**(self.encoder.num_resolutions - i - 1))

        if self.fusion == 'concat':
            for i in range(len(n_embed)):
                self.ms_quantize.append(VectorQuantizer(n_embed[i], embed_dim[i], beta=quant_beta,
                                                remap=remap, sane_index_shape=sane_index_shape, legacy=legacy, init_normal=init_norm)
                                        )
                in_channel = 2*encoder_config["z_channels"][i] if encoder_config["double_z"] else encoder_config["z_channels"][i]
                self.ms_quant_conv.append(torch.nn.Conv2d(in_channel, embed_dim[i], 1))
            embed_dim_sum = sum(embed_dim)
        else:
            self.ms_quantize.append(VectorQuantizer(n_embed[0], embed_dim[0], beta=quant_beta,
                                                remap=remap, sane_index_shape=sane_index_shape, legacy=legacy, init_normal=init_norm)
                                        )
            in_channel = 2 * encoder_config["z_channels"][0] if encoder_config["double_z"] else encoder_config["z_channels"][0]
            self.ms_quant_conv.append(torch.nn.Conv2d(in_channel, embed_dim[0], 1))

            embed_dim_sum = embed_dim[0]
        self.post_quant_conv = torch.nn.Conv2d(embed_dim_sum, decoder_config["z_channels"], 1)

        # share structure
        for i in range(len(n_embed)-1):
            self.upsample.append(nn.ConvTranspose2d(
                embed_dim[0], embed_dim[0], 4, stride=2, padding=1
            ))
            self.shared_post_quant_conv.append(torch.nn.Conv2d(embed_dim[0], encoder_config["z_channels"][0], 1))
            self.shared_decoder.append(Decoder(double_z=False, z_channels=sum(embed_dim[:(i+2)]), resolution=256, in_channels=embed_dim[:(i+2)], 
                     out_ch=embed_dim[0], ch=128, ch_mult=[ 1 ], num_res_blocks=2, attn_resolutions=[2, 4, 8, 16, 32, 64], dropout=0.0))

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        
    def encode(self, x):
        h_ms = self.encoder(x)

        qaunt_ms = []
        emb_loss_ms = []
        info_ms = [[], [], []]
        h_ms = h_ms[::-1]
        prev_h = []
        for ii in range(len(h_ms)):

            if len(prev_h) != 0:
                for j in range(ii):
                    prev_h[j] = self.upsample[ii-1](prev_h[j])
                    prev_h[j] = self.shared_post_quant_conv[ii-1](prev_h[j])
                
                quant = torch.cat((*prev_h[:ii], h_ms[ii]), dim=1)
                quant = self.shared_decoder[ii-1](quant)
                # quant = quant + prev_h
            else:
                quant = h_ms[ii]

            h = self.ms_quant_conv[ii](quant)
            quant, emb_loss, info = self.ms_quantize[ii](h)

            qaunt_ms.append(quant)
            emb_loss_ms.append(emb_loss)
            for jj in range(len(info)):
                info_ms[jj].append(info[jj])
            prev_h.append(quant)

        qaunt_ms = qaunt_ms[::-1]
        # # upsample each resolutions
        for i in range(len(h_ms)):
            for t in range(i):
                qaunt_ms[i] = F.interpolate(qaunt_ms[i], scale_factor=2)

        quant = torch.cat(qaunt_ms, dim=1) # channel-wise concate
        emb_loss = sum(emb_loss_ms)
        return quant, emb_loss, info_ms

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_info=False):
        quant, diff, info = self.encode(input)
        quant_aux = quant.clone()

        # TODO: remove dummy
        quant_aux[:, :-1*self.embed_dim[-1], :, :] = 0
        quant_aux2 = quant.clone()
        quant_aux2[:, self.embed_dim[-1]:, :, :] = 0
        
        dec = self.decode(quant)
        dec_aux = self.decode(quant_aux)
        dec_aux2 = self.decode(quant_aux2)
        
        dec_aux = [dec_aux, dec_aux2]
        
        if self.use_aux_loss:
            return dec, dec_aux, diff, info
        
        if return_info:
            return dec, diff, info
        return dec, diff, info
   
    def get_input(self, batch, num_sampling = None):
        I = batch["image"]
        C = batch["cloth"]
        C_W = self.get_warped_cloth(batch)
        
        if num_sampling is not None:
            I = I[:num_sampling]
            C = C[:num_sampling]
            C_W = C_W[:num_sampling]
                
        return I, C, C_W
        
    def get_loss(self, batch, stage) -> Dict[str, int]:
        I, C, C_W = self.get_input(batch)
        
        x = I
        
        xrec_aux = None
        if self.use_aux_loss:
            xrec, xrec_aux, qloss, _ = self(x)
        else:
            xrec, qloss, _ = self(x)
        
        if stage == "autoencoder":
            # autoencode
            total_loss, log_dict_ae = self.criterion(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", xrec_aux=xrec_aux)
        elif stage == "discriminator":
            # discriminator
            total_loss, log_dict_disc = self.criterion(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        else:
            xrec, qloss, _ = self(x)
            
        return losses
    
    def inference(self):
        pass
    
    def get_image_log(self):
        pass
    
    def save_model(self):
        torch.save(, self.model_path)
    
    def configure_optimizers(self, lr):
        opt_ae = self.optimizer(list(self.encoder.parameters())+
                                list(self.decoder.parameters())+
                                list(self.ms_quantize.parameters())+
                                list(self.ms_quant_conv.parameters())+
                                list(self.post_quant_conv.parameters())+
                                list(self.upsample.parameters())+
                                list(self.shared_decoder.parameters())+
                                list(self.shared_post_quant_conv.parameters()),
                                lr=lr, betas=(0.5, 0.9))

        opt_disc = self.optimizer(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler_ae = LambdaLR(opt_ae, lambda2)
        scheduler_disc = LambdaLR(opt_disc, lambda2)
        
        optimizers = [opt_ae, opt_disc]
        schedulers = [scheduler_ae, scheduler_disc]
        
        return optimizers, schedulers
    
class Lit_vae(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 sampling_step: int,
                 num_sampling: int,
                 model_name: str,
                 model_args: tuple,
                 img2img: bool = True) -> None:
        super().__init__()
        self.lr = lr
        self.sampling_step = sampling_step
        self.num_sampling = num_sampling
        self.img2img = img2img
        
        self.model = getattr(importlib.import_module(__name__), model_name)(**model_args)
        
    def configure_optimizers(self):
        optims, schedulers = self.model.configure_optimizers(self.lr)
        
        return optims, schedulers
    
    def training_step(self, batch, batch_idx):
        losses = self.model.get_loss(batch)
        
        self.logging_loss(losses, "train")
        self.logging_output(batch, "train")
        
        return losses["total"]
    
    def on_train_epoch_end(self):
        self.model.save_model()
        
    def validation_step(self, batch, batch_idx):
        losses = self.model.get_loss(batch)
        
        self.logging_loss(losses, "val")
    
    def test_step(self, batch, batch_idx):
        losses = self.model.get_loss(batch)
        
        self.logging_loss(losses, "test")
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x0_hat = self.predict(batch)
    
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        pass
        
    def predict(self, batch):
        x0_hat = self.model.inference(batch, self.num_sampling, self.img2img)
        
        return x0_hat
        
    def logging_loss(self, losses: Dict[str, int], prefix):
        for key in losses:
            self.log(f'{prefix}/{key}_loss', losses[key], prog_bar=True, sync_dist=True)
            
    def get_grid(self, inputs: Dict[str, torch.Tensor], return_pil=False):        
        for key in inputs:
            image = (inputs[key]/ 2 + 0.5).clamp(0, 1)
            
            if return_pil:
                inputs[key] = self.numpy_to_pil(make_grid(image))
            else:
                inputs[key] = make_grid(image)
        
        return inputs
    
    def sampling(self, batch, prefix="train"):
        outputs = self.model.get_image_log(batch, self.num_sampling)
        
        output_grids = self.get_grid(outputs)
        
        for key in output_grids:
            self.logger.experiment.add_image(f'{prefix}/{key}', output_grids[key], self.current_epoch)
                
    def logging_output(self, batch, prefix="train"):
        if self.global_rank == 0:
            if self.trainer.is_last_batch:
                if self.current_epoch == 0:
                    self.sampling(batch, prefix)
                elif (self.current_epoch + 1) % self.sampling_step == 0:
                    self.sampling(batch, prefix)                