import os
from einops import rearrange
from typing import Optional, Union, List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import make_grid
from torchvision import transforms

from models.Diffusion.Frido.taming.modules.diffusionmodules.model import Decoder
from models.Diffusion.Frido.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from run import instantiate_from_config, get_obj_from_str

from models.base import Module_base, Lit_base

class msvqgan(Module_base):
    def __init__(self, 
                 optim_target: str,
                 criterion_config: tuple,
                 encoder_config: tuple,
                 decoder_config: tuple,
                 embed_dim: list = [4, 4],
                 n_embed: list = [8192, 8192],
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
        super().__init__(model_path)
        self.optimizer = get_obj_from_str(optim_target)
        self.criterion = instantiate_from_config(criterion_config)
        
        self.fusion = fusion
        self.embed_dim = embed_dim
        self.use_aux_loss = use_aux_loss
        self.unsample_type = unsample_type
        
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.ms_quantize = nn.ModuleList()
        self.ms_quant_conv = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.shared_decoder = nn.ModuleList()
        self.shared_post_quant_conv = nn.ModuleList()
        
        assert len(n_embed) == encoder_config.params['multiscale'], 'multiscale mode. dim of n_embed is incorrect.'
        assert len(n_embed) == len(embed_dim), 'multiscale mode. dim of n_embed is incorrect.'

        self.res_list = []
        for i in range(self.encoder.multiscale):
            self.res_list.append(self.encoder.resolution / 2**(self.encoder.num_resolutions - i - 1))

        if self.fusion == 'concat':
            for i in range(len(n_embed)):
                self.ms_quantize.append(VectorQuantizer(n_embed[i], embed_dim[i], beta=quant_beta,
                                                remap=remap, sane_index_shape=sane_index_shape, legacy=legacy, init_normal=init_norm)
                                        )
                in_channel = 2*encoder_config.params["z_channels"][i] if encoder_config.params["double_z"] else encoder_config.params["z_channels"][i]
                self.ms_quant_conv.append(torch.nn.Conv2d(in_channel, embed_dim[i], 1))
            embed_dim_sum = sum(embed_dim)
        else:
            self.ms_quantize.append(VectorQuantizer(n_embed[0], embed_dim[0], beta=quant_beta,
                                                remap=remap, sane_index_shape=sane_index_shape, legacy=legacy, init_normal=init_norm)
                                        )
            in_channel = 2 * encoder_config.params["z_channels"][0] if encoder_config.params["double_z"] else encoder_config.params["z_channels"][0]
            self.ms_quant_conv.append(torch.nn.Conv2d(in_channel, embed_dim[0], 1))

            embed_dim_sum = embed_dim[0]
        self.post_quant_conv = torch.nn.Conv2d(embed_dim_sum, decoder_config.params["z_channels"], 1)

        # share structure
        for i in range(len(n_embed)-1):
            self.upsample.append(nn.ConvTranspose2d(
                embed_dim[0], embed_dim[0], 4, stride=2, padding=1
            ))
            self.shared_post_quant_conv.append(torch.nn.Conv2d(embed_dim[0], encoder_config.params["z_channels"][0], 1))
            self.shared_decoder.append(Decoder(double_z=False, z_channels=sum(embed_dim[:(i+2)]), resolution=256, in_channels=embed_dim[:(i+2)], 
                     out_ch=embed_dim[0], ch=128, ch_mult=[ 1 ], num_res_blocks=2, attn_resolutions=[2, 4, 8, 16, 32, 64], dropout=0.0))

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            
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
        
        if num_sampling is not None:
            I = I[:num_sampling]
            C = C[:num_sampling]
                
        return I, C
        
    def get_loss(self, batch, global_step) -> Dict[str, int]:
        I, C = self.get_input(batch)
        
        x = I
        
        xrec_aux = None
        if self.use_aux_loss:
            xrec, xrec_aux, qloss, _ = self(x)
        else:
            xrec, qloss, _ = self(x)
        
        # autoencode
        losses_ae = self.criterion(qloss, x, xrec, 0, global_step,
                                        last_layer=self.get_last_layer(), xrec_aux=xrec_aux)
        # discriminator
        losses_disc = self.criterion(qloss, x, xrec, 1, global_step,
                                        last_layer=self.get_last_layer())
        
        return losses_ae, losses_disc
    
    def inference(self):
        pass
    
    def get_image_log(self, batch, num_sampling = None):
        I, C = self.get_input(batch, num_sampling)
        
        x = I
        if self.use_aux_loss:
            x_rec, _, _, _ = self(x)
        else:
            x_rec, _, _ = self(x)
        
        return {f"real": x,
                f"fake": x_rec}
         
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

        opt_disc = self.optimizer(self.criterion.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        lambda2 = lambda epoch: 0.95 ** epoch
        
        scheduler_ae = LambdaLR(opt_ae, lambda2)
        scheduler_disc = LambdaLR(opt_disc, lambda2)
        
        optimizers = [opt_ae, opt_disc]
        schedulers = [scheduler_ae, scheduler_disc]
        
        return optimizers, schedulers
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

class Lit_msvqgan(Lit_base):
    def __init__(self,
                 lr: float,
                 model_config,
                 sampling_step: int = 5,
                 num_sampling: int = 20) -> None:
        super().__init__(lr, model_config, sampling_step, num_sampling)
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        optim_ae, optim_disc = self.optimizers()

        losses_ae, losses_disc = self.model.get_loss(batch, self.global_step)
        
        self.toggle_optimizer(optim_ae)
        
        optim_ae.zero_grad()
        self.logging_loss(losses_ae, "train")
        self.logging_output(batch, "train")
        self.manual_backward(losses_ae["total_ae_loss"].requires_grad_(True))
        optim_ae.step()
        
        self.untoggle_optimizer(optim_ae)
        
        self.toggle_optimizer(optim_disc)
        
        optim_disc.zero_grad()
        self.logging_loss(losses_disc, "train")
        self.manual_backward(losses_disc["total_disc_loss"].requires_grad_(True))
        optim_disc.step()
        
        self.untoggle_optimizer(optim_disc)
    
    def validation_step(self, batch, batch_idx):        
        losses_ae, losses_disc = self.model.get_loss(batch, self.global_step)
        
        self.logging_loss(losses_ae, "val")
        self.logging_loss(losses_disc, "val")
    
    def test_step(self, batch, batch_idx):
        losses_ae, losses_disc = self.model.get_loss(batch, self.global_step)
        
        self.logging_loss(losses_ae, "test")
        self.logging_loss(losses_disc, "test")