import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import csv
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import cv2
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc
import json
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Hyperparameters
seed_value = 42
BATCH_SIZE = 64  # Updated to match DataLoader batch size
PATCH_SIZE = 256  # Updated to match the resized image dimensions

# Fixing random seed for reproducibility
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)

holdout_dir = '/export/fhome/vlia/HelicoDataSet/HoldOut/'

patient_ids = []
diag = []
data = pd.read_csv('/export/fhome/vlia/HelicoDataSet/PatientDiagnosis.csv')
for index, row in data.iterrows():
  patient_ids.append(row[0])
  if row[1]=="NEGATIVA":
      diag.append(-1)
  else:
      diag.append(1)

transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),  # Resize images to PATCH_SIZE x PATCH_SIZE
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

class ImageDataset(Dataset):
    def __init__(self, base_dir, patient_ids, diag, transform=None):
        self.base_dir = base_dir
        self.patient_ids = patient_ids
        self.diagnosis = diag
        self.transform = transform
        self.image_paths = []
        self.image_patch_ids = {}
        self.path_to_patient = {}

        for folder_name in os.listdir(base_dir):
            if folder_name[:-2] in patient_ids:
                folder_path = os.path.join(base_dir, folder_name)
                
                # List image files in the folder
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    
                    if image_path.endswith(('.png', '.jpg', '.jpeg')):
                    # window_id = str((os.path.splitext(image_name)[0]).lstrip('0'))
                        window_id = str(int((os.path.splitext(image_name)[0]).split('_')[0]))
                        patch_id = folder_name[:-2] + "_" + window_id
                        self.image_paths.append(image_path)
                        self.image_patch_ids[image_path] = patch_id  # Store patch ID with image path
                        self.path_to_patient[image_path] = folder_name[:-2]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        patch_id = self.image_patch_ids[image_path]
        patient = self.path_to_patient[image_path]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
          image = self.transform(image)

        return {"image": image, "patch_id": patch_id}

 
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder (no max pooling, using stride-2 convolutions instead)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 256x256x3 -> 128x128x32
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 128x128x32 -> 64x64x64
            nn.ReLU()
        )
        
        # Decoder using ConvTranspose2d for upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 64x64x64 -> 128x128x32
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 128x128x32 -> 256x256x3
            nn.Sigmoid()  # Output layer with Sigmoid activation
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def compute_red_loss(original,reconstructed):
    orig = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    rec = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2HSV)

    orig_hue = orig[:, :, 0]
    rec_hue = rec[:, :, 0]

    mask_1_orig = (orig_hue > 0) & (orig_hue < 20) # & (orig_v > 0)
    mask_2_orig = (orig_hue > 160) & (orig_hue < 180) # & (orig_v > 0)
    mask_1_rec = (rec_hue > 0) & (rec_hue < 20) # & (rec_v > 0)
    mask_2_rec = (rec_hue > 160) & (rec_hue < 180) # & (rec_v > 0)

    # Combine both masks
    combined_mask_orig = mask_1_orig | mask_2_orig
    combined_mask_rec = mask_1_rec | mask_2_rec

    original_count = np.sum(combined_mask_orig)
    reconstructed_count = np.sum(combined_mask_rec)

    red_loss = original_count / (reconstructed_count + 1)

    return red_loss

def evaluate_red_loss(model, dataloader):
    model.eval()
    red_losses = []
    true_labels = []  
    patch_ids = []
    rec_errors = []
    
    with torch.no_grad(): 
        for batch in dataloader:
            images = batch["image"] 
            patches = batch["patch_id"] 
            
            # Pass images through the model to get reconstructions
            reconstructions = model(images)

            images = images.cpu().numpy()
            reconstructions = reconstructions.detach().cpu().numpy()
            
            # Calculate red_loss for each original-reconstruction pair in the batch
            for (original, reconstructed, patch) in zip(images, reconstructions, patches):

                original = np.transpose(original, (1,2,0))
                reconstructed = np.transpose(reconstructed, (1,2,0))
                    
                red_loss_value = compute_red_loss(original, reconstructed)
                rec_error = np.mean((original - reconstructed) ** 2)
                
                red_losses.append(red_loss_value)
                patch_ids.append(patch)
                rec_errors.append(rec_error)

    return red_losses, patch_ids, rec_errors


# MAIN
holdout_dataset = ImageDataset(holdout_dir, patient_ids, diag, transform=transform)
holdout_dataloader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AutoEncoder()

model.load_state_dict(torch.load('/export/fhome/vlia02/MyVirtualEnv/autoencoder64.pth'))

red_losses, patch_ids, rec_errors = evaluate_red_loss(model, holdout_dataloader)


redloss_per_patient = {}

for fred, patch_id in zip(red_losses, patch_ids):
    patient = patch_id.split('_')[0]
    if patient not in redloss_per_patient:
        redloss_per_patient[patient] = [fred]
    else:
        redloss_per_patient[patient].append(fred)


# ROC: 91.1
# Otsu: 106.4
roc_thr = 91.1
otsu_thr = 106.4

patient_roc_preds = {}
patient_otsu_preds = {}

for patient, fred_list in redloss_per_patient.items():
    patient_roc_preds[patient] = []
    patient_otsu_preds[patient] = []

    for v in fred_list:
        if v>=roc_thr:
            patient_roc_preds[patient].append(1)
        else:
            patient_roc_preds[patient].append(-1)
        
        if v>=otsu_thr:
            patient_otsu_preds[patient].append(1)
        else:
            patient_otsu_preds[patient].append(-1)


f1 = '/export/fhome/vlia02/MyVirtualEnv/redloss_per_patient.json'
f2 = '/export/fhome/vlia02/MyVirtualEnv/patient_roc_preds.json'
f3 = '/export/fhome/vlia02/MyVirtualEnv/patient_otsu_preds.json'

with open(f1, 'w') as file:
  json.dump(redloss_per_patient, file)

with open(f2, 'w') as file:
  json.dump(patient_roc_preds, file)

with open(f3, 'w') as file:
  json.dump(patient_otsu_preds, file)