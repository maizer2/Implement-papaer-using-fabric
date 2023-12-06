import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.clip.configuration_clip import CLIPVisionConfig

class InversionAdapter(nn.Module):
    def __init__(self,
                 clip_config: CLIPVisionConfig, 
                 num_encoder_layers, 
                 output_dim, 
                 dropout=0.5, 
                 num_vstar = 16,
                 model_path = None):
        super().__init__()
        self.clip_config = clip_config
        self.num_vstar = num_vstar
        self.encoder_layers = nn.ModuleList([CLIPEncoderLayer(clip_config) for _ in range(num_encoder_layers)])
        self.post_layernorm = nn.LayerNorm(clip_config.hidden_size, eps=clip_config.layer_norm_eps)
        
        self.layers = nn.Sequential(
            nn.Linear(clip_config.hidden_size, clip_config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(clip_config.hidden_size * 4, clip_config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(clip_config.hidden_size * 4, output_dim),
        )

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
            
    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, None, None)
            x = x[0]
        x = x[:, 0, :]
        x = self.post_layernorm(x)
        return self.layers(x)
