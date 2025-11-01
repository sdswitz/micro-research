import os    
import math
import numpy as np
import time

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
# from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
import itertools

from my_diffusion import Diffusion, Sampler
    
image_size=64
evaluation_interval=500
total_timesteps=500
plot_freq_interval=50
num_generations=5
num_training_steps=5
num_input_channels=3
batch_size=64
path_to_generated="generated"    

checkpoint_interval = num_training_steps // 5
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

final_dir = "final_models"
os.makedirs(final_dir, exist_ok=True)

# print("Checkpoint Directory:", checkpoint_dir)

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
    
### Define Basic Image Transformations (From Huggingface Annotated Diffusion) ###
image2tensor = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                # transforms.Lambda(lambda t: (t*2) - 1),
                transforms.Normalize([0.5]*num_input_channels, [0.5]*num_input_channels),
            ])
dataset = ImageFolder("/Users/samswitz/GitHub/micro-research/diffusion/data/celeba", transform=image2tensor)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

model = Diffusion(in_channels=num_input_channels).to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Number of Parameters: {params:,.0f}")

### MODEL TRAINING INPUTS ###
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0005)
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                            num_warmup_steps=2500, 
                                            num_training_steps=num_training_steps)

ddpm_sampler = Sampler(num_training_steps=total_timesteps)

loss_fn = nn.MSELoss()

progress_bar = tqdm(range(num_training_steps))
completed_steps = 0

train = True
while train:
    training_losses = []
    
    if (completed_steps + 1) % checkpoint_interval == 0:
        
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if 'optimizer' in locals() else None,
            "scheduler": scheduler.state_dict() if 'scheduler' in locals() else None,
            "scaler": scaler.state_dict() if 'scaler' in locals() else None,
            "completed_steps": completed_steps+1,
            "pytorch_version": torch.__version__,
        }
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"train_state_{timestamp}.pth")
        torch.save(ckpt, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    for images, _ in trainloader:
        batch_size = images.shape[0]
    
        ### Random Sample T ###
        timesteps = torch.randint(0,total_timesteps,(batch_size,))
    
        ### Get Noisy Images ###
        noisy_images, noise = ddpm_sampler.add_noise(images, timesteps)
    
        ### Get Noise Prediction ###
        noise_pred = model(noisy_images.to(device), timesteps.to(device))

        ### Compute Error ###
        loss = loss_fn(noise_pred, noise.to(device))

        training_losses.append(loss.cpu().item())
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        progress_bar.update(1)
        completed_steps += 1

        if (completed_steps % evaluation_interval == 0):
            loss_mean = np.mean(training_losses)
            print("Training Loss:", loss_mean)
            print("Learning Rate:", optimizer.param_groups[-1]["lr"])

            training_losses = []
            
        if completed_steps >= num_training_steps:
            print("Training Completed!!!")
            final_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if 'optimizer' in locals() else None,
                "scheduler": scheduler.state_dict() if 'scheduler' in locals() else None,
                "scaler": scaler.state_dict() if 'scaler' in locals() else None,
                "completed_steps": completed_steps+1,
                "pytorch_version": torch.__version__,
            }
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            final_state_path = os.path.join(final_dir, f"train_state_{timestamp}.pth")
            torch.save(final_state, final_state_path)
            print(f"Saved final state: {final_state_path}")
            
            train = False
            break
# train()
