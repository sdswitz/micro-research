import os    
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import itertools

class Sampler:
    def __init__(self, num_training_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_training_steps = num_training_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.beta_schedule = self.linear_beta_schedule()
        
        self.alpha = 1 - self.beta_schedule
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=-1)
        
    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_training_steps)
    
    def _repeated_unsqueeze(self, target, tensor):
        while target.dim() > tensor.dim():
            tensor = tensor.unsqueeze(-1)
        return tensor
        
    def add_noise(self, image, timesteps):
        
        batch_size, c, h, w = image.shape
        
        device = image.device
        
        alpha_cumulative_prod_timesteps = self.alpha_cumulative_prod[timesteps].to(device)

        mean_coeff = alpha_cumulative_prod_timesteps ** 0.5
        
        var_coeff = (1 - alpha_cumulative_prod_timesteps) ** 0.5
        
        mean_coeff = self._repeated_unsqueeze(image, mean_coeff)
        var_coeff = self._repeated_unsqueeze(image, var_coeff)
        
        noise = torch.randn_like(image)
        
        noisy_image = (mean_coeff * image) + (var_coeff * noise)
        
        return noisy_image, noise
    
    def remove_noise(self, image, timesteps, predicted_noise):
        
        b, c, h, w = image.shape
        
        device = image.device
        
        equal_to_zero_mask = (timesteps == 0)
        
        beta_t = self.beta_schedule[timesteps].to(device)
        alpha_t = self.alpha[timesteps].to(device)
        alpha_cumulative_prod_t = self.alpha_cumulative_prod[timesteps].to(device)
        alpha_cumulative_prod_t_prev = self.alpha_cumulative_prod[timesteps-1].to(device)
        
        alpha_cumulative_prod_t_prev[equal_to_zero_mask] = 1.0
        
        noise = torch.randn_like(image)
        
        variance = beta_t * (1 - alpha_cumulative_prod_t_prev) / (1 - alpha_cumulative_prod_t)
        
        variance = self._repeated_unsqueeze(image, variance)
        
        sigma_tz = variance ** 0.5 * noise
        
        noise_coefficient = beta_t / (1 - alpha_cumulative_prod_t) ** 0.5
        noise_coefficient = self._repeated_unsqueeze(image, noise_coefficient)
        
        reciprocal_root_a_t = alpha_t ** -0.5
        reciprocal_root_a_t = self._repeated_unsqueeze(image, reciprocal_root_a_t)
        
        mean = reciprocal_root_a_t  * (image - (noise_coefficient * predicted_noise))
        
        denoised = mean + sigma_tz
        
        return denoised
            

class SelfAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 num_heads=12,
                 attn_p=0,
                 proj_p=0):
        
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        
        self.attn_p = attn_p
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        
        batch_size, seq_len, embed_dim = x.shape
        
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p)
        
        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
        
        
class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 mlp_ratio=4,
                 mlp_p=0):
        
        super().__init__()
        
        self.fc1 = nn.Linear(in_channels, in_channels * mlp_ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(in_channels * mlp_ratio, in_channels)
        self.drop2 = nn.Dropout(mlp_p)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x
        
class TransformerBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 num_heads=4,
                 mlp_ratio=2,
                 proj_p=0,
                 attn_p=0,
                 mlp_p=0):
        
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        
        self.attn = SelfAttention(in_channels=in_channels,
                                  num_heads=num_heads,
                                  attn_p=attn_p,
                                  proj_p=proj_p,)
        
        self.norm2 = nn.LayerNorm(in_channels, eps=1e-6)
        self.mlp = MLP(in_channels=in_channels,
                       mlp_ratio=mlp_ratio,
                       mlp_p=mlp_p)
        
    def forward(self, x):
        
        batch_size, channels, height, width = x.shape
        
        x = x.reshape(batch_size, channels, height * width).permute(0,2,1)
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        x = x.permute(0,2,1).reshape(batch_size, channels, height, width)
        
        return x

class SinusoidalTimeEmbedding(nn.Module):
    
    def __init__(self, time_embed_dim, scaled_time_embed_dim):
        super().__init__()
        
        self.inv_freq = nn.Parameter(1.0 / (10000 ** (torch.arange(0, time_embed_dim, 2).float() / time_embed_dim)), requires_grad=False)
        
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, scaled_time_embed_dim),
                                      nn.SiLU(),
                                      nn.Linear(scaled_time_embed_dim, scaled_time_embed_dim),
                                      nn.SiLU())
        
    def forward(self, timsteps):
        
        timestem_freqs = timesteps.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        
        embeddings = torch.cat([torch.sin(timestem_freqs), torch.cos(timestem_freqs)], dim=-1)
        
        embeddings = self.time_mlp(embeddings)
        
        return embeddings
    

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, groupnorm_num_groups, time_embed_dim):
        super().__init__()
        
        self.time_expand = nn.Linear(time_embed_dim, out_channels)
        
        self.groupnorm1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        
        self.groupnorm2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        
        self.resize_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_embeddings):
        
        residual_connection = x
        
        time_embed = self.time_expand(time_embeddings)
        
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # x = x + time_embed.unsqueeze(-1).unsqueeze(-1)
        x = x + time_embed.reshape((*time_embed.shape, 1, 1))
        
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        x = x + self.resize_channels(residual_connection)
        
        return x
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
        )
    
    def forward(self, x):
        return self.upsample(x)
    
class UNET(nn.Module):
    def __init__(self,
                 in_channels=3,
                 start_dim=64,
                 dim_mults=(1,2,4),
                 residual_blocks_per_group=1,
                 groupnorm_num_groups=16,
                 time_embed_dim=128):
        
        super().__init__()
        
        self.input_image_channels = in_channels
        
        channel_sizes = [start_dim * i for i in dim_mults]
        starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]
        
        self.encoder_config = []
        
        for idx, d in enumerate(channel_sizes):
            
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(((d, d), "residual"))
            
            self.encoder_config.append(((d, d), "downsample"))
            
            self.encoder_config.append((d, "attention"))
            
            if idx < len(channel_sizes) - 1:
                self.encoder_config.append(((d, channel_sizes[idx+1]), "residual"))
                
        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))
        
        out_dim = ending_channel_size
        reversed_encoder_config = self.encoder_config[::-1]
        
        self.decoder_config = []
        for idx, (metadata, l_type) in enumerate(reversed_encoder_config):
            
            if l_type != "attention":
                enc_in_channels, enc_out_channels = metadata
                self.decoder_config.append(
                    (
                        (out_dim+enc_out_channels, enc_in_channels), "residual"
                    )
                )
                
                if l_type == "downsample":
                    self.decoder_config.append(((enc_in_channels, enc_in_channels), "upsample"))
                    
                out_dim = enc_in_channels
            
            else:
                in_channels = metadata
                self.decoder_config.append((in_channels, "attention"))
                
        self.decoder_config.append(((starting_channel_size * 2, starting_channel_size), "residual"))
        
        # Model Build
        
        self.conv_in_proj = nn.Conv2d(self.input_image_channels, starting_channel_size, kernel_size=3, padding="same")
        
        self.encoder = nn.ModuleList()
        for metadata, l_type in self.encoder_config:
            if l_type == "residual":
                in_channels, out_channels = metadata
                self.encoder.append(
                    ResidualBlock(
                        in_channels, out_channels, groupnorm_num_groups, time_embed_dim
                    )
                )
            elif l_type == "downsample":
                in_channels, out_channels = metadata
                self.encoder.append(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=2, padding=1
                    )
                )
            elif l_type == "attention":
                in_channels = metadata
                self.encoder.append(TransformerBlock(in_channels))
                
        self.bottleneck = nn.ModuleList()
        for (in_channels, out_channels), _ in self.bottleneck_config:
            self.bottleneck.append(
                ResidualBlock(
                    in_channels, out_channels, groupnorm_num_groups, time_embed_dim
                )
            )
            
        self.decoder = nn.ModuleList()
        for metadata, l_type in self.decoder_config:
            if l_type == "residual":
                in_channels, out_channels = metadata
                self.decoder.append(ResidualBlock(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  groupnorm_num_groups=groupnorm_num_groups,
                                                  time_embed_dim=time_embed_dim))
            elif l_type == "upsample":
                in_channels, out_channels = metadata
                self.decoder.append(UpSampleBlock(in_channels=in_channels,
                                                  out_channels=out_channels))
            elif l_type == "attention":
                in_channels = metadata
                self.decoder.append(TransformerBlock(in_channels))
                
        self.conv_out_proj = nn.Conv2d(starting_channel_size, self.input_image_channels, kernel_size=3, padding="same")
    
    def forward(self, x, time_embeddings):
        
        residuals = []
        
        x = self.conv_in_proj(x)
        residuals.append(x)
        
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x, time_embeddings)
                residuals.append(x)
            elif isinstance(module, nn.Conv2d):
                x = module(x)
                residuals.append(x)
            else:
                x = module(x)
        
        for module in self.bottleneck:
            x = module(x, time_embeddings)
        
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                residual_tensor = residuals.pop()
                x = torch.cat([x, residual_tensor], dim=1)
                x = module(x, time_embeddings)
            else:
                x = module(x)
                
        x = self.conv_out_proj(x)
        return x
        
class Diffusion(nn.Module):
    def __init__(self,
                 in_channels=3,
                 start_dim=64,
                 dim_mults=(1,2,4,4),
                 residual_blocks_per_group=1,
                 groupnorm_num_groups=16,
                 time_embed_dim=128,
                 time_embed_dim_ratio=2):
        
        super().__init__()
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.dim_mults = dim_mults
        self.residual_blocks_per_group = residual_blocks_per_group
        self.groupnorm_num_groups = groupnorm_num_groups

        self.time_embed_dim = time_embed_dim
        self.scaled_time_embed_dim = int(time_embed_dim * time_embed_dim_ratio)

        self.sinusoid_time_embeddings = SinusoidalTimeEmbedding(time_embed_dim=self.time_embed_dim,
                                                                scaled_time_embed_dim=self.scaled_time_embed_dim)

        self.unet = UNET(in_channels=in_channels, 
                         start_dim=start_dim, 
                         dim_mults=dim_mults, 
                         residual_blocks_per_group=residual_blocks_per_group, 
                         groupnorm_num_groups=groupnorm_num_groups,  
                         time_embed_dim=self.scaled_time_embed_dim)

    def forward(self, noisy_inputs, timesteps):

        ### Embed the Timesteps ###
        timestep_embeddings = self.sinusoid_time_embeddings(timesteps)
        
        ### Pass Images + Time Embeddings through UNET ###
        noise_pred = self.unet(noisy_inputs, timestep_embeddings)

        return noise_pred