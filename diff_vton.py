import importlib, os, argparse
from collections import namedtuple
from omegaconf import OmegaConf
from typing import Optional, Union

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from cp_dataset import CPDataset, CPDataLoader


torch.set_float32_matmul_precision('medium')


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                        type=str, 
                        # default="configs/diffusion/Diffusers_LDM.yaml",
                        # default="configs/diffusion/Diffusers_UnconditionalLDM.yaml",
                        default="configs/diffusion/Diffusers_DDIM.yaml",
                        # default="configs/diffusion/Diffusers_DDPM.yaml",
                        # default="configs/diffusion/DDPM.yaml",
                        # default="configs/vae/VQ-VAE.yaml",
                        # default="configs/ae/Unet.yaml",
                        # default="configs/ae/ConvAE.yaml",
                        # default="configs/ae/MLPAE.yaml",
                        # default="configs/gan/WGAN_GP.yaml",
                        # default="configs/gan/WGAN.yaml",
                        # default="configs/gan/CGAN.yaml",
                        # default="configs/gan/DCGAN.yaml",
                        # default="configs/gan/VanilaGAN.yaml",
                        # default="configs/cnn/ResNet.yaml",
                        # default="configs/cnn/VGGNet.yaml",
                        # default="configs/cnn/AlexNet.yaml",
                        # default="configs/cnn/LeNet5.yaml", 
                        # default="configs/mlp/MLP.yaml",
                        help="Path of model config file.")
    parser.add_argument("--out_path",
                        type=str,
                        default="./",
                        help="Generation model output path")
    
    # Dataset opt
    parser.add_argument("--data_name", 
                        type=str, 
                        default="CIFAR10", 
                        help="Torchvision dataset name.")
    parser.add_argument("--data_path", 
                        type=str, 
                        # default="data",
                        default="data/VITON/",
                        help="Path of dataset.")
    parser.add_argument("--image_mean", 
                        type=float, 
                        default=0.5, 
                        help="image mean.")
    parser.add_argument("--image_std", 
                        type=float, 
                        default=0.5, 
                        help="image std.")
    
    
    # cp-vton arguments
    parser.add_argument("--stage", default = "TOM", help="GMM or TOM")
    parser.add_argument("--model_name", default="DDIM", help="DDPM, DDIM, StableDiffusion, etc.")
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    
    
    # dataset arguments
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--image_size", type=tuple, default = (256, 192), help="Image size shape like (height, width)")
    parser.add_argument("--img_size", type=tuple, default = (256, 192), help="Image size shape like (height, width)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_sampling', type=int, default=3)
    
    
    
    # lightning arguments
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--max_epochs", type=int, default = 3_125_000)
    parser.add_argument('--ckpt_path', type=str, default=None, help='model checkpoint for initialization')
    
    
    # diffusion arguments
    parser.add_argument("--eta", default=0.0)
    parser.add_argument("--sampling_step", default=50)
    parser.add_argument("--num_train_steps", default=1000)
    parser.add_argument("--num_inference_steps", default=50)
    
    opt = parser.parse_args()
    return opt


def get_dataloader(opt, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((opt.image_mean, ), (opt.image_std, )),
                                        transforms.Resize(opt.image_size, antialias=True)
                                        ])
    train_dataset = getattr(importlib.import_module("torchvision.datasets"), opt.data_name)(root=opt.data_path, 
                                                                                            download=True, 
                                                                                            transform=transform)
    
    train_dataset, val_dataset = data.random_split(train_dataset, [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)])
    test_dataset = getattr(importlib.import_module("torchvision.datasets"), opt.data_name)(root=opt.data_path, 
                                                                                            download=True,
                                                                                            train=False,
                                                                                            transform=transform)
    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, 
                                   num_workers=opt.num_workers
                                   )
    
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.num_workers
                                 )
    
    test_loader = data.DataLoader(test_dataset, 
                            batch_size=opt.batch_size, 
                            num_workers=opt.num_workers
                            )
    
    return train_loader, val_loader, test_loader
    

def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def get_log_path(config):
    module_name = config.model.target.split(".")[1]
    model_name = config.model.params.model_name
    return os.path.join("checkpoints", module_name, model_name)
    
    
if __name__ == "__main__":
    opt = get_opt()
    config = OmegaConf.load(opt.config)
    # train_loader, val_loader, test_loader = get_dataloader(opt)
    
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset).get_loader()
    
    from networks import Lit_VTON
    model = Lit_VTON(opt)
    
    trainer = pl.Trainer(max_epochs=opt.max_epochs,
                         default_root_dir=get_log_path(config),
                         log_every_n_steps=7,
                         strategy='ddp_find_unused_parameters_true',
                        #  devices=[0, 1, 2, 3, 4, 5, 6]
                        #  devices=[0]
                         # callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
                         )
    
    if not opt.inference:
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    # val_dataloaders=val_loader,
                    ckpt_path=opt.ckpt_path)
        
    #     trainer.test(model=model,
    #                  dataloaders=test_loader)
        
    else:
        opt.ckpt_path = "checkpoints/diffusion/DDIM/lightning_logs/TOM_final.ckpt"
        model.eval()
        trainer.predict(model=model, 
                        # dataloaders=test_loader,
                        dataloaders=train_loader,
                        ckpt_path=opt.ckpt_path)
        