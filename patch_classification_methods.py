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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


# Hyperparameters
seed_value = 42
BATCH_SIZE = 64  # Updated to match DataLoader batch size
PATCH_SIZE = 256  # Updated to match the resized image dimensions

# Fixing random seed for reproducibility
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)

annotated_dir = 'C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/Annotated/'

patch_ids = []
patient_ids = []
presence = []
data = pd.read_excel('C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/HP_WSI-CoordAnnotatedPatches.xlsx')
for index, row in data.iterrows():
  if row[7]!=0:
    patch_id = f"{row[0]}_{row[2]}"
    patch_ids.append(patch_id)
    patient_ids.append(row[0])
    presence.append(row[7])

transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),  # Resize images to PATCH_SIZE x PATCH_SIZE
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

class ImageDataset(Dataset):
    def __init__(self, base_dir, patient_ids, patch_ids, presence, transform=None):
        self.base_dir = base_dir
        self.patient_ids = patient_ids
        self.patch_ids = patch_ids
        self.labels = presence
        self.transform = transform
        self.image_paths = []
        self.image_patch_ids = {}

        # Create a mapping between patch_id and label and patch_id and patient_id
        self.patch_to_label = {f"{patch_id}": label for patch_id, label in zip(patch_ids, presence)}
        self.patch_to_patient = {f"{patch_id}": patient for patch_id, patient in zip(patch_ids, patient_ids)}

        for folder_name in os.listdir(base_dir):
            if folder_name[:-2] in patient_ids:
                folder_path = os.path.join(base_dir, folder_name)
                
                # List image files in the folder
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    
                    if image_path.endswith(('.png', '.jpg', '.jpeg')): 
                      window_id = str(int((os.path.splitext(image_name)[0]).split('_')[0]))
                      patch_id = folder_name[:-2] + "_" + window_id
  
                      if patch_id in self.patch_ids:
                        self.image_paths.append(image_path)
                        self.image_patch_ids[image_path] = patch_id  # Store patch ID with image path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        patch_id = self.image_patch_ids[image_path]
        label = self.patch_to_label[patch_id]
        patient = self.patch_to_patient[patch_id]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
          image = self.transform(image)

        return {"image": image, "label": label, "patch_id": patch_id}

 
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 256x256x3 -> 128x128x32
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 128x128x32 -> 64x64x64
            nn.ReLU()
        )
        
        # Decoder
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

    mask_1_orig = (orig_hue > 0) & (orig_hue < 20) 
    mask_2_orig = (orig_hue > 160) & (orig_hue < 180)
    mask_1_rec = (rec_hue > 0) & (rec_hue < 20)
    mask_2_rec = (rec_hue > 160) & (rec_hue < 180)

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
            labels = batch["label"] 
            patches = batch["patch_id"] 
            
            # Pass images through the model to get reconstructions
            reconstructions = model(images)

            images = images.cpu().numpy()
            reconstructions = reconstructions.detach().cpu().numpy()
            
            # Calculate red_loss for each original-reconstruction pair in the batch
            for (original, reconstructed, patch, label) in zip(images, reconstructions, patches, labels):

                original = np.transpose(original, (1,2,0))
                reconstructed = np.transpose(reconstructed, (1,2,0))
                    
                red_loss_value = compute_red_loss(original, reconstructed)
                rec_error = np.mean((original - reconstructed) ** 2)
                
                red_losses.append(red_loss_value)
                true_labels.append(label)
                patch_ids.append(patch)
                rec_errors.append(rec_error)
                
    return red_losses, true_labels, patch_ids, rec_errors

def store_results(red_losses,patch_ids):
  redloss_dic = {}

  for redl, patchid in zip(red_losses,patch_ids):
    redloss_dic[patchid] = redl
    
  return redloss_dic

def compute_roc_threshold(red_losses, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, red_losses)

    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    closest_idx = np.argmin(distances)
    optimal_threshold = thresholds[closest_idx]

    return optimal_threshold, fpr, tpr

def compute_otsu_threshold(red_losses):
    red_loss_values = np.array(red_losses)
    red_loss_values = red_loss_values.astype(np.uint8)

    otsu_threshold, _ = cv2.threshold(red_loss_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return otsu_threshold


# MAIN
annotated_dataset = ImageDataset(annotated_dir, patient_ids, patch_ids, presence, transform=transform)
annotated_dataloader = DataLoader(annotated_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AutoEncoder()

model.load_state_dict(torch.load('C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/final_autoencoder.pth', map_location=torch.device('cpu')))

red_losses, labels, patch_ids, rec_errors = evaluate_red_loss(model, annotated_dataloader)

patient_ids = [patch.split('_')[0] for patch in patch_ids]

# Initialize K-Fold
k_folds = 5
kf = StratifiedGroupKFold(n_splits=k_folds, shuffle=False)

labels_list = [tensor.item() for tensor in labels]
labels = np.array(labels_list)
patient_ids = np.array(patient_ids)
red_losses = np.array(red_losses)
rec_errors = np.array(rec_errors)
patch_ids = np.array(patch_ids)

val_recalls_roc = []
val_recalls_otsu = []
roc_thresholds = []
otsu_thresholds = []

val_accuracies_roc = []
val_accuracies_otsu = []

conf_matrix_roc = np.zeros((2, 2))
conf_matrix_otsu = np.zeros((2, 2))

plt.figure(figsize=(10, 8))

folds_train = {}
folds_test = {}
red_loss_dic = {p:l for p,l in zip(patch_ids,red_losses)}
roc_pred = {}
otsu_pred = {}

for i, (train_index, test_index) in enumerate(kf.split(patch_ids, labels, patient_ids)):
    # Store train and test indices of each fold
    for patch in patch_ids[train_index]:
        folds_train[patch] = i
    for patch in patch_ids[test_index]:
        folds_test[patch] = i

    roc_threshold, fpr_roc, tpr_roc = compute_roc_threshold(red_losses[train_index], labels[train_index])
    val_predictions_roc_train = [1 if fred >= roc_threshold else -1 for fred in red_losses[train_index]]
    val_predictions_roc = [1 if fred >= roc_threshold else -1 for fred in red_losses[test_index]]
    roc_auc = auc(fpr_roc, tpr_roc)
    val_recall_roc = recall_score(labels[test_index], val_predictions_roc)
    val_recalls_roc.append(val_recall_roc)
    roc_thresholds.append(roc_threshold)
    val_accuracies_roc.append(accuracy_score(labels[test_index], val_predictions_roc))

    roc_pred_train = {p:pred for p,pred in zip(patch_ids[train_index], val_predictions_roc_train)}
    roc_pred_test = {p:pred for p,pred in zip(patch_ids[test_index], val_predictions_roc)}
    roc_pred.update(roc_pred_train)
    roc_pred.update(roc_pred_test)

    cm = confusion_matrix(labels[test_index], val_predictions_roc, labels=[-1, 1])
    conf_matrix_roc += cm

    plt.plot(fpr_roc, tpr_roc, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    otsu_threshold = compute_otsu_threshold(red_losses[train_index])
    val_predictions_otsu_train = [1 if fred >= otsu_threshold else -1 for fred in red_losses[train_index]]
    val_predictions_otsu = [1 if fred >= otsu_threshold else -1 for fred in red_losses[test_index]]
    val_recall_otsu = recall_score(labels[test_index], val_predictions_otsu)
    val_recalls_otsu.append(val_recall_otsu)
    otsu_thresholds.append(otsu_threshold)

    val_accuracies_otsu.append(accuracy_score(labels[test_index], val_predictions_otsu))

    cm2 = confusion_matrix(labels[test_index], val_predictions_otsu, labels=[-1, 1])
    conf_matrix_otsu += cm2

    otsu_pred_train = {p:pred for p,pred in zip(patch_ids[train_index], val_predictions_otsu_train)}
    otsu_pred_test = {p:pred for p,pred in zip(patch_ids[test_index], val_predictions_otsu)}
    otsu_pred.update(otsu_pred_train)
    otsu_pred.update(otsu_pred_test)

    print(f"Optimal ROC threshold: {roc_threshold}, ROC Validation Recall: {val_recall_roc}")
    print(f"Optimal Otsu threshold: {otsu_threshold}, Otsu Validation Recall: {val_recall_otsu}")

best_roc_threshold = np.mean(roc_thresholds)
best_otsu_threshold = np.mean(otsu_thresholds)
print(f"\nOptimal ROC Threshold across folds: {best_roc_threshold:.4f}")
print(f"Optimal Otsu Threshold across folds: {best_otsu_threshold:.4f}")

print(f"\nROC Validation Recall across folds: {np.mean(val_recalls_roc)}")
print(f"Otsu Validation Recall across folds: {np.mean(val_recalls_otsu)}")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

# ROC plot
plt.title('ROC Curves for Each Fold')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_roc, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', ax=plt.gca(), values_format='.0f')
plt.title('Confusion Matrix (ROC Threshold)')
plt.show()

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_otsu, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', ax=plt.gca(), values_format='.0f')
plt.title('Confusion Matrix (Otsu Threshold)')
plt.show()



# SVM CLASSIFIER
# Define the parameter grid for the SVM
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 1, 10]
}

svm_cm = np.zeros((2, 2), dtype=int)
svm_cm_adjusted = np.zeros((2, 2), dtype=int)

accuracies_svm = []
recalls_svm = []
adjusted_recalls_svm = []

for i, (train_index, test_index) in enumerate(kf.split(patch_ids, labels, patient_ids)):
    print("Training on fold", i)

    X_train = np.column_stack((red_losses[train_index], rec_errors[train_index]))
    y_train = labels[train_index]

    X_test = np.column_stack((red_losses[test_index], rec_errors[test_index]))
    y_test = labels[test_index]

    # Create a pipeline for scaling and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ])

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, 
                               scoring='recall',
                               cv=3,
                               n_jobs=-1)


    grid_search.fit(X_train, y_train)

    # Get the best estimator from the grid search
    best_model = grid_search.best_estimator_
    best_svm_params = grid_search.best_params_
    best_svm_params = {key.replace('svm__', ''): value for key, value in best_svm_params.items()}

    print(f"Best parameters for fold {i}: {grid_search.best_params_}")

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)

    decision_scores = best_model.decision_function(X_test)

    # Adjust the threshold to increase recall
    custom_threshold = -0.5  # Shift threshold to favor the positive class
    y_pred_adjusted = np.where(decision_scores > custom_threshold, 1, -1)

    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted, labels=[-1, 1])
    svm_cm += cm
    svm_cm_adjusted += cm_adjusted

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    accuracies_svm.append(acc)
    recall = recall_score(y_test, y_pred)
    recalls_svm.append(recall)
    recall_adjusted = recall_score(y_test, y_pred_adjusted)
    adjusted_recalls_svm.append(recall_adjusted)

# Print average metrics across folds
print(f"\nAverage Accuracy across {kf.n_splits} folds: {np.mean(accuracies_svm):.4f}")
print(f"Average Recall across {kf.n_splits} folds: {np.mean(recalls_svm):.4f}")
print(f"Adjusted Recall across {kf.n_splits} folds: {np.mean(adjusted_recalls_svm):.4f}")

disp_total = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=['Negative', 'Positive'])
disp_total.plot(cmap='Blues', values_format='d')
plt.title("SVM Confusion Matrix")
plt.show()

disp_total = ConfusionMatrixDisplay(confusion_matrix=svm_cm_adjusted, display_labels=['Negative', 'Positive'])
disp_total.plot(cmap='Blues', values_format='d')
plt.title("Adjusted SVM Confusion Matrix")
plt.show()


# KNN CLASSIFIER
# List of k values to evaluate
k_values = [5, 10, 50, 100, 150, 200]

# Initialize lists to store metrics for each k
knn_metrics = {k: {'accuracy': 0, 'recall': 0, 'cf':np.zeros((2,2),dtype=int)} for k in k_values}

for k in k_values:
    print(f"\nEvaluating k-NN with k={k}")
    cm_knn = np.zeros((2, 2), dtype=int)
    knn_accs = []
    knn_rec = []
    
    for i, (train_index, test_index) in enumerate(kf.split(patch_ids, labels, patient_ids)):
        print(f"Training on fold {i} for k={k}")
        
        X_train = np.column_stack((red_losses[train_index], rec_errors[train_index]))
        y_train = labels[train_index]

        X_test = np.column_stack((red_losses[test_index], rec_errors[test_index]))
        y_test = labels[test_index]

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize and train the k-NN model
        knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
        cm_knn += cm
        
        acc = accuracy_score(y_test, y_pred)
        knn_accs.append(acc)
        recall = recall_score(y_test, y_pred)
        knn_rec.append(recall)

    # Average metrics across folds for this k
    avg_acc = np.mean(knn_accs)
    avg_recall = np.mean(knn_rec)
    knn_metrics[k]['accuracy'] = avg_acc
    knn_metrics[k]['recall'] = avg_recall
    knn_metrics[k]['cf'] = cm_knn
    print(f"Average Accuracy for k={k}: {avg_acc:.4f}")
    print(f"Average Recall for k={k}: {avg_recall:.4f}")

best_k_by_recall = max(knn_metrics, key=lambda k: knn_metrics[k]['recall'])
print(f"Best k by recall: {best_k_by_recall}")
print(f"KNN Accuracy: {knn_metrics[best_k_by_recall]['accuracy']:.4f}")
print(f"KNN Recall: {knn_metrics[best_k_by_recall]['recall']:.4f}")

disp_total = ConfusionMatrixDisplay(confusion_matrix=knn_metrics[best_k_by_recall]['cf'], display_labels=['Negative', 'Positive'])
disp_total.plot(cmap='Blues', values_format='d')
plt.title(f"KNN Confusion Matrix (k={best_k_by_recall})")
plt.show()


# HYBRID MODEL
final_acc = []
final_recall = []

print(red_losses)
print(labels)

cm_hybrid = np.zeros((2,2), dtype=int)

for i, (train_index, test_index) in enumerate(kf.split(patch_ids, labels, patient_ids)):

    roc_threshold, fpr_roc, tpr_roc = compute_roc_threshold(red_losses[train_index], labels[train_index])
    val_predictions_roc = [1 if fred >= roc_threshold else -1 for fred in red_losses[test_index]]

    otsu_threshold = compute_otsu_threshold(red_losses[train_index])
    val_predictions_otsu = [1 if fred >= otsu_threshold else -1 for fred in red_losses[test_index]]

    X_train = np.column_stack((red_losses[train_index], rec_errors[train_index]))
    y_train = labels[train_index]

    X_test = np.column_stack((red_losses[test_index], rec_errors[test_index]))
    print('ok')
    y_test = labels[test_index]

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # Use transform instead of fit_transform for test data

    svm_model = SVC(**best_svm_params)
    svm_model.fit(X_train, y_train)
    
    svm_pred = svm_model.predict(X_test)

 
    knn_model = KNeighborsClassifier(n_neighbors=best_k_by_recall, weights='distance')
    knn_model.fit(X_train, y_train)
    
    knn_pred = knn_model.predict(X_test)

    predictions_stack = np.vstack([np.array(val_predictions_roc), np.array(val_predictions_otsu), np.array(svm_pred), np.array(knn_pred)])

    # Majority voting
    final_predictions = []

    for i1, i2, i3, i4 in zip(val_predictions_roc, val_predictions_otsu, list(svm_pred), list(knn_pred)):
        votes = [i1, i2, i3, i4]
    
        # Count the occurrences of each class
        count_1 = votes.count(1)
        count_neg1 = votes.count(-1)
        
        # Majority vote: choose the class with the most votes
        if count_1 > count_neg1:
            final_predictions.append(1)  # Majority vote is 1
        else:
            final_predictions.append(-1)  # Majority vote is -1

    final_predictions = np.array(final_predictions)

    final_acc.append(accuracy_score(labels[test_index], final_predictions))
    final_recall.append(recall_score(labels[test_index], final_predictions))

    cm = confusion_matrix(labels[test_index], final_predictions, labels=[-1,1])

    cm_hybrid += cm

print("Accuracy for Hybrid model across folds:", np.mean(final_acc))
print("Recall for Hybrid model across folds:", np.mean(final_recall))

disp_total = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid, display_labels=['Negative', 'Positive'])
disp_total.plot(cmap='Blues', values_format='d')
plt.title(f"Hybrid Confusion Matrix)")
plt.show()


f1 = 'C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/redloss_dic.json'
f2 = 'C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/roc_pred_dic.json'
f3 = 'C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/otsu_pred_dic.json'
f4 = 'C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/folds_train.json'
f5 = 'C:/Users/Usuario/Desktop/Artificial Intelligence/5 cuatri/Vision and Learning/Challenge 2/folds_test.json'

with open(f1, 'w') as file:
  json.dump(red_loss_dic, file)

with open(f2, 'w') as file:
  json.dump(roc_pred, file)

with open(f3, 'w') as file:
  json.dump(otsu_pred, file)

with open(f4, 'w') as file:
  json.dump(folds_train, file)

with open(f5, 'w') as file:
  json.dump(folds_test, file)