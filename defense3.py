import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


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
TARGET_CLASSES = list(range(20))  # All 20 classes (CIFAR10 + Camouflaged digits)
PATCH_SIZE = 5  # Size of the trigger mask
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.1
SAVE_DIR = "neural_cleanse"
os.makedirs(SAVE_DIR, exist_ok=True)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 20)
model.load_state_dict(torch.load("resnet18_exp3_stealth.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
base_ds = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
base_loader = DataLoader(base_ds, batch_size=BATCH_SIZE, shuffle=False)

def apply_trigger(x, trigger, mask, position=(0, 0)):
    b, c, h, w = x.shape
    px, py = position
    ps = trigger.shape[-1]
    x_new = x.clone()
    patch_region = (1 - mask) * x[:, :, px:px+ps, py:py+ps] + mask * trigger
    x_new[:, :, px:px+ps, py:py+ps] = patch_region
    return x_new

mask_norms = []
all_masks = []

for target in tqdm(TARGET_CLASSES, desc="Optimizing trigger masks per class"):
    trigger = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE, device=DEVICE, requires_grad=True)
    mask = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([trigger, mask], lr=LR)

    for epoch in range(EPOCHS):
        for x, _ in base_loader:
            x = x.to(DEVICE)
            y_target = torch.full((x.size(0),), target, dtype=torch.long, device=DEVICE)

            x_adv = apply_trigger(x, trigger, mask.sigmoid(), position=(0, 0))
            preds = model(x_adv)

            loss_ce = F.cross_entropy(preds, y_target)
            loss_reg = torch.norm(mask.sigmoid().view(-1), p=1)
            loss = loss_ce + 1e-3 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        m_final = mask.sigmoid().cpu().numpy()
        norm = np.linalg.norm(m_final.ravel(), ord=1)
        mask_norms.append(norm)
        all_masks.append(m_final)
        np.save(f"{SAVE_DIR}/mask_class_{target}.npy", m_final)

mask_norms = np.array(mask_norms)
median = np.median(mask_norms)
mad = np.median(np.abs(mask_norms - median))
z_scores = np.abs(mask_norms - median) / (mad + 1e-6)
suspected = np.where(z_scores > 2.5)[0]

plt.figure(figsize=(10, 5))
plt.bar(TARGET_CLASSES, mask_norms, color='gray')
for i in suspected:
    plt.bar(i, mask_norms[i], color='red')
plt.title("Neural Cleanse Trigger Mask Norms")
plt.xlabel("Target Class")
plt.ylabel("L1 Norm of Optimized Mask")
plt.xticks(TARGET_CLASSES)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/mask_norms_bar.png")
plt.close()

np.save(f"{SAVE_DIR}/suspected_targets.npy", suspected)
print(f"\n? Neural Cleanse Completed.")
print(f"Suspected Hijack Classes (by high anomaly score): {suspected}")
print(f"Trigger masks, scores, and plots saved to '{SAVE_DIR}/'")
