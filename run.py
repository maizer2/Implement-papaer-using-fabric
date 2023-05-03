import importlib, os

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from configure import get_opt
from models.mlp.MLP import LitMultiLayerPerceptron

torch.set_float32_matmul_precision('medium')
        
        
def get_dataloader(opt, transform=None):
    
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, )),
                                        transforms.Resize(opt.img_size, antialias=True)
                                        ])
        
    train_dataset = datasets.MNIST("data", download=True, transform=transform)
    
    train_dataset, val_dataset = data.random_split(train_dataset, [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)])
    test_dataset = datasets.MNIST("data", download=True, train=False, transform=transform)
    
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
    
    
def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)
    
    
if __name__ == "__main__":
    opt = get_opt()
        
    train_loader, val_loader, test_loader = get_dataloader(opt)
    
    model = get_obj_from_str(opt.model_name)()
    
    trainer = pl.Trainer(max_epochs=200,
                         default_root_dir=opt.log_path,
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