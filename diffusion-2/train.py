import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model import UNet, Sampler

import pandas as pd

train_data_path = './data/train'
test_data_path = './data/test'

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

train_data = datasets.CIFAR100(root=train_data_path, train=True, download=True, transform=transform)
test_data = datasets.CIFAR100(root=test_data_path, train=False, download=True, transform=transform)
print("successfully loaded data")

EPOCHS = 10
lr = 2e-4

batch_size = 128
input_channels = 3
num_classes = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_interval = 100
test_eval_interval = 500
save_interval = int(EPOCHS / 4)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) #, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) #, num_workers=2)

sampler = Sampler()
model = UNet(in_channels=input_channels, out_channels=input_channels, num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"\nModel initialized.")
print(f"    number of parameters: {sum(p.numel() for p in model.parameters()):,.0f}")

def evaluate(model, dataloader, sampler, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            t = torch.randint(0, 1000, (images.shape[0],), device=device)
            noisy_images, noise = sampler.add_noise(images, t)
            pred_noise = model(noisy_images, t, labels)
            total_loss += F.mse_loss(pred_noise, noise).item()
            num_batches += 1
    model.train()
    return total_loss / num_batches

train_losses = []
test_losses = []

model.train()
for i in range(EPOCHS):
    epoch_loss = 0
    num_batches = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        t = torch.randint(0, 1000, (images.shape[0],), device=device).long()
        
        noisy_image, noise = sampler.add_noise(images, t)
        pred_noise = model(noisy_image, t, labels)
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    
    print(f"Epoch {i+1}/{EPOCHS}, Train Loss: {avg_epoch_loss:.4f}")
    
    if (i + 1) % save_interval == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{i+1}.pt")
    
    if (i + 1) % test_eval_interval == 0:
        test_loss = evaluate(model, test_loader, sampler, device)
        test_losses.append((i + 1, test_loss))
        print(f"Epoch {i+1}/{EPOCHS}, Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        

test_loss_dict = dict(test_losses)
loss_history_df = pd.DataFrame({
    'epoch': list(range(1, EPOCHS + 1)),
    'train_loss': train_losses,
    'test_loss': [test_loss_dict.get(e, None) for e in range(1, EPOCHS + 1)]
})