import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os
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
BATCH_SIZE = 64
NUM_CLASSES = 20
SAVE_DIR = "activation_clustering_defense"
os.makedirs(SAVE_DIR, exist_ok=True)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("resnet18_exp3_stealth.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

activations = []

def hook_fn(module, input, output):
    activations.append(output.detach().squeeze(-1).squeeze(-1).cpu())

model.avgpool.register_forward_hook(hook_fn)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
camo = torch.load("camouflaged_dataset/camouflaged_dataset.pt")
camo_images = (camo["images"] - 0.5) / 0.5
camo_labels = camo["labels"] + 10
camo_set = CamouflageDataset(camo_images, camo_labels, transform)
dataset = ConcatDataset([cifar, camo_set])
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Extracting activations...")
feature_bank, label_bank = [], []
with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        _ = model(x)
        feature_bank.append(activations.pop())
        label_bank.append(y)

features = torch.cat(feature_bank, dim=0).numpy()
labels = torch.cat(label_bank, dim=0).numpy()

poison_indices = []

for cls in range(10, NUM_CLASSES):
    class_idx = np.where(labels == cls)[0]
    class_feats = features[class_idx]

    if len(class_feats) < 10:
        continue

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    pred = kmeans.fit_predict(class_feats)
    silhouette = silhouette_score(class_feats, pred)

    counts = np.bincount(pred)
    minority_cluster = np.argmin(counts)
    minority_idx = class_idx[pred == minority_cluster]

    poison_indices.extend(minority_idx.tolist())

    plt.hist(pred, bins=2)
    plt.title(f"Class {cls} Clusters | Silhouette = {silhouette:.2f}")
    plt.xlabel("Cluster ID")
    plt.ylabel("Samples")
    plt.savefig(f"{SAVE_DIR}/cluster_class_{cls}.png")
    plt.close()

poison_mask = np.zeros(len(labels), dtype=bool)
poison_mask[poison_indices] = True
np.save(f"{SAVE_DIR}/poison_mask.npy", poison_mask)

print(f"Activation Clustering Completed.")
print(f"Detected {np.sum(poison_mask)} suspicious samples.")
print(f"Saved mask and plots to '{SAVE_DIR}/'")
