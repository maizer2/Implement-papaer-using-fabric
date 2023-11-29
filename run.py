import importlib, os, argparse
from omegaconf import OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

import torch
from torch.utils import data
from torchvision import transforms

torch.set_float32_matmul_precision('medium')


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
                        choices=["DDPM", "diffusers_DDPM", "diffusers_DDIM", "diffusers_LDM", "diffusers_text_to_LDM"])
    parser.add_argument("--dataset", required=True,
                        choices=["MNIST", "CIFAL10"])
    
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
    
    base_logger_path = os.path.join(logger_config.logger_path, opt.model_type, opt.model_name, opt.dataset)
    
    model_config.params["model_name"] = opt.model_name
    model_config.params.model_args.unet_config["sample_size"] = (data_config.height, data_config.width)
    
    logger_config.logger_path = os.path.join(base_logger_path, str(data_config.height))
    
    data_config["name"] = opt.dataset
    data_config["data_path"] = os.path.join(data_config.data_path, data_config.name)
    
    return model_config, logger_config, lightning_config, data_config


def get_dataloader(data_config, transform=None):
    if transform is None:
        transform = [transforms.ToTensor(),
                     transforms.Normalize((0.5, ), (0.5, )),
                     transforms.Resize((data_config.height, data_config.width), antialias=True)]
        
        if data_config.name == "MNIST":
            transform.insert(0, transforms.Grayscale(num_output_channels=3))
            
        transform = transforms.Compose(transform)
        
    train_dataset = getattr(importlib.import_module("torchvision.datasets"), data_config.name)(root=data_config.data_path, 
                                                                                                download=True, 
                                                                                                transform=transform)
    
    train_dataset, val_dataset = data.random_split(train_dataset, [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)])
    
    test_dataset = getattr(importlib.import_module("torchvision.datasets"), data_config.name)(root=data_config.data_path, 
                                                                                                download=True,
                                                                                                train=False,
                                                                                                transform=transform)
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
    

def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)

    
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
        