import importlib, os, argparse
from collections import namedtuple
from omegaconf import OmegaConf

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


torch.set_float32_matmul_precision('medium')
        
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", 
                        action="store_true",
                        help="When inferring the model")
    parser.add_argument("--config", 
                        type=str, 
                        default="configs/diffusion/DDPM.yaml",
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
    parser.add_argument("--data_path", 
                        type=str, 
                        default="data", 
                        help="Path of dataset.")
    parser.add_argument("--ckpt_path", 
                        type=str, 
                        default=None,
                        # default="checkpoints/cnn/vggnet/lightning_logs/version_0/checkpoints/epoch=2-step=750.ckpt",
                        help="Path of ckpt.")
    parser.add_argument("--num_workers", 
                        type=tuple, 
                        default=6, 
                        help="Number of DataLoader worker.")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=64, 
                        help="Batch size.")
    parser.add_argument("--max_epochs", 
                        type=int, 
                        default=100, 
                        help="Epoch lenghts.")
    parser.add_argument("--out_path",
                        type=str,
                        default="./",
                        help="Generation model output path")
    
    opt = parser.parse_args()
    return opt


def get_dataloader(opt, config, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((config.data.image_mean, ), (config.data.image_std, )),
                                        transforms.Resize(config.model.params.model_args.image_size, antialias=True)
                                        ])
        
    train_dataset = datasets.MNIST(opt.data_path, download=True, transform=transform)
    
    train_dataset, val_dataset = data.random_split(train_dataset, [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)])
    test_dataset = datasets.MNIST(opt.data_path, download=True, train=False, transform=transform)
    
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
    train_loader, val_loader, test_loader = get_dataloader(opt, config)
    
    model = instantiate_from_config(config.model)
    
    trainer = pl.Trainer(max_epochs=opt.max_epochs,
                         default_root_dir=get_log_path(config),
                         strategy='ddp_find_unused_parameters_true'
                         # callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
                         )
    
    if not opt.inference:
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=opt.ckpt_path)
        
        trainer.test(model=model,
                     dataloaders=test_loader)
        
    else:
        model.eval()
        trainer.predict(model=model, 
                        dataloaders=test_loader,
                        ckpt_path=opt.ckpt_path)
        