import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class CamouflageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, torch.Tensor):
            if img.max() > 1.0:  # If in [0,255], scale to [0,1]
                img = img.float() / 255.0
            img = ToPILImage()(img)    # Convert to PIL for transform
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.images)

class TensorLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        label = torch.tensor(label, dtype=torch.long)
        return img, label
    def __len__(self):
        return len(self.dataset)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 20
TOP_K = 1
OUTLIER_PERCENT = 0.05
BATCH_SIZE = 64
SAVE_DIR = "spectral_defense_output"
os.makedirs(SAVE_DIR, exist_ok=True)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("resnet18_exp3_stealth.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

feature_bank = []
label_bank = []

def hook_fn(module, input, output):
    feature_bank.append(output.detach().squeeze(-1).squeeze(-1).cpu())

model.avgpool.register_forward_hook(hook_fn)


TARGET_SIZE = 224

transform = Compose([
    Resize(TARGET_SIZE),          # Resize to 224x224
    ToTensor(),                   # Convert PIL Image to Tensor (C, H, W)
    Normalize((0.5,), (0.5,)),    # Normalize to [-1, 1]
])

cifar = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
camo = torch.load("camouflaged_dataset/camouflaged_dataset.pt")
camo_images = (camo["images"] - 0.5) / 0.5
camo_labels = camo["labels"] + 10
camo_set = CamouflageDataset(camo_images, camo_labels, transform)
combined_set = ConcatDataset([cifar, camo_set])

loader = DataLoader(combined_set, batch_size=BATCH_SIZE, shuffle=False)

print("Extracting features...")
with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        _ = model(x)
        label_bank.append(y.cpu())

features = torch.cat(feature_bank, dim=0).numpy()
labels = torch.cat(label_bank, dim=0).numpy()

poison_indices = []

for cls in range(NUM_CLASSES):
    cls_indices = np.where(labels == cls)[0]
    if len(cls_indices) < 10:
        continue
    cls_features = features[cls_indices]
    
    pca = PCA(n_components=TOP_K)
    proj = pca.fit_transform(cls_features)
    magnitudes = np.linalg.norm(proj, axis=1)
    threshold = np.percentile(magnitudes, 100 * (1 - OUTLIER_PERCENT))

    flagged = cls_indices[magnitudes > threshold]
    poison_indices.extend(flagged.tolist())

    plt.figure(figsize=(6, 4))
    plt.hist(magnitudes, bins=30, alpha=0.7, label='Proj. Magnitudes')
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title(f"PCA Spectrum - Class {cls}")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/pca_class_{cls}.png")
    plt.close()

poison_mask = np.zeros(len(labels), dtype=bool)
poison_mask[poison_indices] = True
np.save(os.path.join(SAVE_DIR, "poison_mask.npy"), poison_mask)

print(f"Spectral Signature Defense completed.")
print(f"Detected {np.sum(poison_mask)} suspected poisoned samples.")
print(f"Saved plots and mask to '{SAVE_DIR}'")