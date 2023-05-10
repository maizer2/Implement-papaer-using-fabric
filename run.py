import importlib, os, argparse
from omegaconf import OmegaConf

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


torch.set_float32_matmul_precision('medium')
        
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                        type=str, 
                        default="configs/VanilaGAN.yaml",
                        # default="configs/ResNet.yaml",
                        # default="configs/VGGNet.yaml",
                        # default="configs/AlexNet.yaml",
                        # default="configs/LeNet5.yaml", 
                        # default="configs/MLP.yaml",
                        help="Path of model config file.")
    parser.add_argument("--data_path", 
                        type=str, 
                        default="data", 
                        help="Path of dataset.")
    parser.add_argument("--log_path", 
                        type=str, 
                        default="checkpoints/gan/vanilagan",
                        # default="checkpoints/cnn/resnet",
                        # default="checkpoints/cnn/vggnet",
                        # default="checkpoints/cnn/alexnet",
                        # default="checkpoints/cnn/lenet5",
                        # default="checkpoints/mlp",
                        help="Path of lightning logs.")
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
                        default=200, 
                        help="Epoch lenghts.")
    
    opt = parser.parse_args()

    os.makedirs(opt.log_path, exist_ok=True)
    return opt


def get_dataloader(opt, config, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((config["image_mean"], ), (config["image_std"], )),
                                        transforms.Resize(config["image_size"], antialias=True)
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
    
    
if __name__ == "__main__":
    opt = get_opt()
    config = OmegaConf.load(opt.config)
    
    train_loader, val_loader, test_loader = get_dataloader(opt, config.data)
    
    model = instantiate_from_config(config.model)
    
    trainer = pl.Trainer(max_epochs=opt.max_epochs,
                         default_root_dir=opt.log_path,
                         num_nodes=1,
                         strategy='ddp_find_unused_parameters_true'
                        #  callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
                         )
    
    trainer.fit(model=model, 
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=opt.ckpt_path
                )
    
    trainer.test(model=model,
                 dataloaders=test_loader
                 )
