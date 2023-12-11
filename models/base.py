import importlib, os
from typing import Optional, Union, List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.optim.adam import Adam
import lightning.pytorch as pl

from torchvision.utils import make_grid

from run import instantiate_from_config, get_obj_from_str

class Module_base(nn.Module):
    def __init__(self, 
                 model_path: str = None
                 ):
        super().__init__()
        self.model_path = model_path
            
    def forward(self):
        pass
   
    def get_input(self):
        pass
        
    def get_loss(self) -> Dict[str, int]:
        pass
    
    def inference(self):
        pass
    
    def get_image_log(self):
        pass
        
    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
    
    def configure_optimizers(self):
        pass

class Lit_base(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 model_config,
                 sampling_step: int = 5,
                 num_sampling: int = 20) -> None:
        super().__init__()
        self.lr = lr
        self.sampling_step = sampling_step
        self.num_sampling = num_sampling
        
        self.model = instantiate_from_config(model_config)
        
    def configure_optimizers(self):
        optims, schedulers = self.model.configure_optimizers(self.lr)
        
        return optims, schedulers
    
    def training_step(self, batch, batch_idx):
        losses = self.model.get_loss(batch)
        
        self.logging_loss(losses, "train")
        self.logging_output(batch, "train")
        
        return losses["train/total_loss"]
    
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
            
    def predict(self, batch):
        x0_hat = self.model.inference(batch, self.num_sampling)
        
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
        outputs = self.model.get_image_log(batch, prefix, self.num_sampling)
        
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
