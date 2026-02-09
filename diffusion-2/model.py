import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()
            
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        t_emb_proj = self.emb_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb_proj
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.nin_shortcut(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)
        # freqs = torch.tensor(10000 ** (-(2 * torch.arange(0, emb_dim//2)) / emb_dim))
        freqs = (10000 ** (-(2 * torch.arange(0, emb_dim//2)) / emb_dim)).clone()
        self.register_buffer('freqs', freqs)
        
    def forward(self, t):
        t = t.float()
        emb = t.unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = self.linear2(F.silu(self.linear1(emb)))
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, fused_attn=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)
        self.fused_attn = fused_attn
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.norm(x)
        qkv = qkv.reshape(b, c, h * w).permute(0, 2, 1)
        qkv = self.qkv(qkv).reshape(b, h * w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0)
        
        if self.fused_attn:
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(b, h * w, c)
            out = self.proj_out(out).permute(0, 2, 1).reshape(b, c, h, w)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
            
        return x + out


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, emb_dim=256):
        super().__init__()
        self.t_emb = TimestepEmbedding(emb_dim)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.encoder = nn.ModuleList([
            # stage 1
            ResBlock(base_channels, base_channels, emb_dim),
            ResBlock(base_channels, base_channels, emb_dim),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            
            # stage 2
            ResBlock(base_channels, base_channels * 2, emb_dim),
            ResBlock(base_channels * 2, base_channels * 2, emb_dim),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            
            # stage 3
            ResBlock(base_channels * 2, base_channels * 2, emb_dim),
            ResBlock(base_channels * 2, base_channels * 2, emb_dim),
        ])
        
        self.bottleneck = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels * 2, emb_dim),
            AttentionBlock(base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2, emb_dim),
        ])
        
        self.decoder = nn.ModuleList([
            # stage 3
            ResBlock(base_channels * 4, base_channels * 2, emb_dim),
            ResBlock(base_channels * 4, base_channels * 2, emb_dim),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            
            # stage 2
            ResBlock(base_channels * 4, base_channels * 2, emb_dim),
            ResBlock(base_channels * 4, base_channels * 2, emb_dim),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            
            # stage 1
            ResBlock(base_channels * 3, base_channels, emb_dim),
            ResBlock(base_channels * 2, base_channels, emb_dim),
        ])
        
        self.output_head = nn.ModuleList([
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        ])
        
    def forward(self, x, t):
        t_emb = self.t_emb(t)
        x = self.conv1(x)
        
        skips = []
        for block in self.encoder:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
                skips.append(x)
            else:
                x = block(x)
            
        for block in self.bottleneck:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
                
        for block in self.decoder:
            if isinstance(block, ResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, t_emb)
            else:
                x = block(x)
        
        x = self.output_head[0](x)
        x = self.output_head[1](x)
        x = self.output_head[2](x)
        return x


class Sampler(nn.Module):
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, num_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_cumprod", torch.cumprod(self.alphas, dim=0))
    
    def _repeated_unsqueeze(self, x, target_dim):
        while x.dim() < target_dim.dim():
            x = x.unsqueeze(1)
        return x
    
    def add_noise(self, inputs, timesteps):
        alpha_timesteps = self.alpha_cumprod[timesteps]
        
        mu_coeff = alpha_timesteps ** 0.5
        sigma_coeff = (1 - alpha_timesteps) ** 0.5
        
        mu_coeff = self._repeated_unsqueeze(mu_coeff, inputs)
        sigma_coeff = self._repeated_unsqueeze(sigma_coeff, inputs)
        
        noise = torch.randn_like(inputs)
        noisy_inputs = mu_coeff * inputs + sigma_coeff * noise
        return noisy_inputs, noise
    
    def remove_noise(self, x_t, timestep, pred_noise):
        equal_zero_mask = (timestep == 0)
        
        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alpha_cumprod[timestep]
        alpha_cumprod_prev = self.alpha_cumprod[timestep - 1]
        alpha_cumprod_prev[equal_zero_mask] = 1.0
        
        noise = torch.randn_like(x_t)
        
        var = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
        var = self._repeated_unsqueeze(var, x_t)
        
        sigma_t_z = noise * var ** 0.5
        sigma_t_z[equal_zero_mask] = 0.0
        
        noise_coeff = beta_t / (1 - alpha_cumprod_t) ** 0.5
        noise_coeff = self._repeated_unsqueeze(noise_coeff, x_t)
        
        reciprocal_root_a_t = alpha_t ** -0.5
        reciprocal_root_a_t = self._repeated_unsqueeze(reciprocal_root_a_t, x_t)
        
        mean = denoised = reciprocal_root_a_t * (x_t - noise_coeff * pred_noise)
        denoised = mean + sigma_t_z
        return denoised
        
        
        
