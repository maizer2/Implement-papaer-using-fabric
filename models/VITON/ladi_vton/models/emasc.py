from typing import List

import torch
import torch.nn as nn


class EMASC(nn.Module):
    """
    EMASC: Enhanced Mask-Aware Skip Connections
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 int_layers: List[int] = None,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1,
                 type: str = 'nonlinear',
                 model_path: str = None):
        super().__init__()
        if int_layers is None:
            int_layers = [i + 1 for i in range(len(in_channels))]
        else:
            if len(in_channels) != len(int_layers):
                raise Exception("Not match lenght.")
            
        self.int_layers = int_layers
        
        if type == 'linear':  # Linear EMASC
            self.conv = nn.ModuleList()
            for in_ch, out_ch in zip(in_channels, out_channels):
                self.conv.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=True, padding=padding, stride=stride))
            self.apply(self._init_weights)
        elif type == 'nonlinear':  # Nonlinear EMASC
            self.conv = nn.ModuleList()
            for in_ch, out_ch in zip(in_channels, out_channels):
                adapter = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, bias=True, padding=padding, stride=stride),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, bias=True, padding=padding, stride=stride),
                )
                self.conv.append(adapter)
        else:
            raise NotImplementedError(f"EMASC type {type} is not implemented.")

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
            
    def forward(self, x: list):
        for i in range(len(x)):
            x[i] = self.conv[i](x[i])
        return x

    def _init_weights(self, w):  # Zero initialization
        if isinstance(w, nn.Conv2d):
            w.weight.data.fill_(0.00)
            w.bias.data.fill_(0.00)
