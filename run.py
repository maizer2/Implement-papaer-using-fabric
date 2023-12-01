import importlib, os, argparse
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

import torch
from torch.utils import data
from torchvision import transforms

from torchvision.datasets import MNIST

torch.set_float32_matmul_precision('medium')

def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true",
                        help="When inferring the model")
    parser.add_argument("--config", type=str, 
                        help="Path of model config file.")
    parser.add_argument("--ckpt_path", type=str,
                        help="Path of ckpt.")
    
    parser.add_argument("--model_type", required=True,
                        choices=["ae", "cnn", "diffusion", "gan", "mlp", "vae"])
    parser.add_argument("--model_name", required=True,
                        choices=["DDPM", "diffusers_DDPM", "diffusers_DDIM", "diffusers_PNDM", 
                                 "diffusers_LDM", "diffusers_StableDiffusion", "diffusers_ControlNet_with_StableDiffusion"])
    parser.add_argument("--dataset", required=True,
                        choices=["MNIST", "CIFAR10"])
    
    opt = parser.parse_args()
    return opt

def check_opt(opt): 
    return opt

def get_config(opt):
    if opt.config is None:
        config = OmegaConf.load(os.path.join("configs", opt.model_type, opt.dataset, f"{opt.model_name}.yaml"))
    else:
        config = OmegaConf.load(opt.config)
        
    model_config, logger_config, lightning_config, data_config = config.model, config.logger, config.lightning, config.dataset
    
    model_config.params["model_name"] = opt.model_name
    
    try:
        model_config.params.model_args.unet_config["sample_size"] = (data_config.height, data_config.width)
    except:
        pass
    
    logger_config.logger_path = os.path.join(logger_config.logger_path, opt.model_type, 
                                             opt.model_name, opt.dataset, str(data_config.height))
    
    return model_config, logger_config, lightning_config, data_config

def transform_init(datasets, transform):
    for dataset in datasets:
        dataset.transform = transform
        
def get_dataloader(data_config, transform=None):
    train_dataset = instantiate_from_config(data_config.train)
    val_dataset = instantiate_from_config(data_config.val)
    test_dataset = instantiate_from_config(data_config.test)
    
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, )),
                                        transforms.Resize((data_config.height, data_config.width), antialias=True)])
        
    
    transform_init([train_dataset, val_dataset, test_dataset], transform)
    
    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=data_config.batch_size, 
                                   num_workers=data_config.num_workers,
                                   drop_last=True
                                   )
    
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=data_config.batch_size,
                                 num_workers=data_config.num_workers,
                                 drop_last=True
                                 )
    
    test_loader = data.DataLoader(test_dataset, 
                                  batch_size=data_config.batch_size, 
                                  num_workers=data_config.num_workers,
                                  drop_last=True
                                  )
                            
    
    return train_loader, val_loader, test_loader
    
if __name__ == "__main__":
    opt = get_opt()
    
    model_config, logger_config, lightning_config, data_config = get_config(opt)
    
    train_loader, val_loader, test_loader = get_dataloader(data_config)
    
    model = instantiate_from_config(model_config)
    
    logger = TensorBoardLogger(logger_config.logger_path)
    
    trainer = pl.Trainer(logger=logger,
                        #  callbacks=[
                        #      EarlyStopping(**lightning_config.earlystop_params),
                        #      LearningRateMonitor(**lightning_config.monitor_params)
                        #      ],
                         **lightning_config.trainer,
                         )
    
    if not opt.inference:
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=opt.ckpt_path)
        
    #     trainer.test(model=model,
    #                  dataloaders=test_loader)
        
    # else:
    #     trainer.predict(model=model.eval(), 
    #                     dataloaders=test_loader,
    #                     # dataloaders=train_loader,
    #                     ckpt_path=opt.ckpt_path)
        