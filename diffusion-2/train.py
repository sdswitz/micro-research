import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from model import UNet, Sampler

import pandas as pd

train_data_path = './data/train'
test_data_path = './data/test'

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

train_data = datasets.CIFAR10(root=train_data_path, train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root=test_data_path, train=False, download=True, transform=transform)
print("successfully loaded data")

EPOCHS = 400
lr = 2e-4

batch_size = 128
input_channels = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_interval = 100
test_eval_interval = 10
save_interval = int(EPOCHS / 4)
ema_decay = 0.9999
sample_interval = 20
num_sample_images = 16
snapshot_dir = "./samples"
os.makedirs(snapshot_dir, exist_ok=True)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) #, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) #, num_workers=2)

sampler = Sampler().to(device)
model = UNet(in_channels=input_channels, out_channels=input_channels, base_channels=128, emb_dim=256)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
ema_model = copy.deepcopy(model).to(device)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad_(False)

print(f"\nModel initialized.")
print(f"    number of parameters: {sum(p.numel() for p in model.parameters()):,.0f}")

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for key in ema_params:
            ema_params[key].mul_(decay).add_(model_params[key], alpha=1.0 - decay)

        ema_buffers = dict(ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for key in ema_buffers:
            ema_buffers[key].copy_(model_buffers[key])

def evaluate(model, dataloader, sampler, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            t = torch.randint(0, 1000, (images.shape[0],), device=device)
            noisy_images, noise = sampler.add_noise(images, t)
            pred_noise = model(noisy_images, t)
            total_loss += F.mse_loss(pred_noise, noise).item()
            num_batches += 1
    model.train()
    return total_loss / num_batches

@torch.no_grad()
def sample_images(model, sampler, device, num_images=16, image_size=32, channels=3):
    model.eval()
    x_t = torch.randn(num_images, channels, image_size, image_size, device=device)

    for step in reversed(range(sampler.num_steps)):
        timestep = torch.full((num_images,), step, device=device, dtype=torch.long)
        pred_noise = model(x_t, timestep)
        x_t = sampler.remove_noise(x_t, timestep, pred_noise)

    x_0 = x_t.clamp(-1, 1)
    x_0 = (x_0 + 1) / 2
    return x_0

train_losses = []
test_losses = []

model.train()
for i in range(EPOCHS):
    epoch_loss = 0
    num_batches = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {i+1}/{EPOCHS}", leave=False)
    for images, _ in train_pbar:
        images = images.to(device)
        t = torch.randint(0, 1000, (images.shape[0],), device=device).long()
        
        noisy_image, noise = sampler.add_noise(images, t)
        pred_noise = model(noisy_image, t)
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema_model, model, ema_decay)
        
        epoch_loss += loss.item()
        num_batches += 1
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    
    print(f"Epoch {i+1}/{EPOCHS}, Train Loss: {avg_epoch_loss:.4f}")
    
    if (i + 1) % save_interval == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i + 1,
            },
            f"checkpoint_epoch_{i+1}.pt",
        )
    
    if (i + 1) % test_eval_interval == 0:
        test_loss = evaluate(ema_model, test_loader, sampler, device)
        test_losses.append((i + 1, test_loss))
        print(f"Epoch {i+1}/{EPOCHS}, Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

    if (i + 1) % sample_interval == 0:
        samples = sample_images(
            ema_model,
            sampler,
            device=device,
            num_images=num_sample_images,
            image_size=32,
            channels=input_channels,
        )
        grid = make_grid(samples, nrow=4)
        save_image(grid, os.path.join(snapshot_dir, f"epoch_{i+1:04d}.png"))
        print(f"Saved sample snapshot: {os.path.join(snapshot_dir, f'epoch_{i+1:04d}.png')}")
        

test_loss_dict = dict(test_losses)
loss_history_df = pd.DataFrame({
    'epoch': list(range(1, EPOCHS + 1)),
    'train_loss': train_losses,
    'test_loss': [test_loss_dict.get(e, None) for e in range(1, EPOCHS + 1)]
})
loss_history_df.to_csv("loss_history.csv", index=False)
