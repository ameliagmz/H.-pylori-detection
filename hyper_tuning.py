# === Necessary Imports ===
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import product
import torch.nn as nn

# === DATA LOADING AND TRANSFORMATION ===
patient_ids = []
healthy_ids = []

with open('/export/fhome/vlia02/MyVirtualEnv/PatientDiagnosis.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        patient_ids.append(row[0])
        if row[1] == "NEGATIVA":
            healthy_ids.append(row[0])

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),         
])

class ImageDataset(Dataset):
    def __init__(self, base_dir, healthy_ids, patient_ids, transform=None):
        self.base_dir = base_dir
        self.healthy_ids = healthy_ids
        self.patient_ids = patient_ids
        self.transform = transform
        self.image_paths = []

        for folder_name in os.listdir(base_dir):
            if folder_name[:-2] in healthy_ids or folder_name[:-2] not in patient_ids:
                folder_path = os.path.join(base_dir, folder_name)
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if image_path.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

base_dir = '/export/fhome/vlia/HelicoDataSet/CrossValidation/Cropped/'
dataset = ImageDataset(base_dir, healthy_ids, patient_ids, transform=transform)

# === MODEL DEFINITION ===
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === TRAINING FUNCTION ===
def train_autoencoder(num_epochs, model, optimizer, train_loader, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    model = model.to(device)
    log_dict = {'train_loss_per_epoch': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for features in train_loader:
            features = features.to(device)
            logits = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        log_dict['train_loss_per_epoch'].append(avg_epoch_loss)
        print(f'Epoch: {epoch + 1}/{num_epochs} | Avg Loss: {avg_epoch_loss:.4f}')
    return log_dict

# === GRID SEARCH IMPLEMENTATION ===
learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [64, 128]
weight_decays = [0, 1e-5, 1e-4]

def train_with_config(config, train_loader):
    lr = config["lr"]
    batch_size = config["batch_size"]
    weight_decay = config["weight_decay"]

    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_dict = train_autoencoder(num_epochs=10, model=model, optimizer=optimizer, train_loader=train_loader)
    return log_dict

def grid_search():
    results = []
    best_config = None
    best_loss = float("inf")

    for lr, bs, wd in product(learning_rates, batch_sizes, weight_decays):
        config = {"lr": lr, "batch_size": bs, "weight_decay": wd}
        print(f"Training with config: {config}")
        subset_size = int(0.5 * len(dataset))
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        subset = Subset(dataset, indices)
        train_loader = DataLoader(subset, batch_size=bs, shuffle=True, pin_memory=True)
        log_dict = train_with_config(config, train_loader)
        final_loss = log_dict['train_loss_per_epoch'][-1]
        results.append({"config": config, "final_loss": final_loss, "log_dict": log_dict})
        if final_loss < best_loss:
            best_loss = final_loss
            best_config = config
    return results, best_config, best_loss

# Run grid search
results, best_config, best_loss = grid_search()

with open("best_config.json", "w") as f:
    json.dump({"best_config": best_config, "best_loss": best_loss}, f, indent=4)

# Plot training loss curve for each combination
plt.figure(figsize=(10, 6))
for r in results:
    plt.plot(r['log_dict']['train_loss_per_epoch'], label=str(r['config']))
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Curves for Hyperparameter Combinations")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("training_loss_curves.png")
plt.show()

# Train with best configuration
final_train_loader = DataLoader(dataset, batch_size=best_config["batch_size"], shuffle=True, pin_memory=True)
final_model = AutoEncoder().to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])

final_log_dict = train_autoencoder(num_epochs=20, model=final_model, optimizer=final_optimizer, train_loader=final_train_loader)
torch.save(final_model.state_dict(), "final_autoencoder.pth")

# Compare 10 original images with their reconstructions
final_model.eval()
original_images = []
reconstructed_images = []
with torch.no_grad():
    for i, image in enumerate(dataset):
        if i >= 10:
            break
        image = image.to(device).unsqueeze(0)
        reconstructed = final_model(image)
        original_images.append(image.cpu().squeeze(0).numpy().transpose(1, 2, 0))
        reconstructed_images.append(reconstructed.cpu().squeeze(0).numpy().transpose(1, 2, 0))

for i in range(10):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original_images[i])
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(reconstructed_images[i])
    plt.axis("off")
    plt.show()
