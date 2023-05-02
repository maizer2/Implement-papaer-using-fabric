import importlib

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from configure import get_opt

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)()
    
if __name__ == "__main__":
    opt = get_opt()
    
    dataset = datasets.MNIST("data", download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize((0.5, ), (0.5 ,)),
                                                                                  transforms.Resize(opt.img_size)
                                                                                  ]))
    train_loader = DataLoader(dataset, 
                              batch_size=opt.batch_size, 
                              num_workers=opt.num_workers
                              )
    
    trainer = pl.Trainer()
    trainer.fit(model=get_obj_from_str(opt.model_name), train_dataloaders=train_loader)