import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
MODEL_PATH = "resnet18_exp3_stealth.pth"
NUM_CLASSES = 20  # 10 CIFAR + 10 Camouflaged digits

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

camo = torch.load("camouflaged_dataset/camouflaged_dataset.pt")
images = camo["images"]
labels = camo["labels"] + 10
images = (images - 0.5) / 0.5
camo_ds = CamouflageDataset(images, labels, transform)
camo_loader = DataLoader(camo_ds, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
all_targets = []

print("\n=== Sample Predictions ===")
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(camo_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

        if batch_idx < 2:
            for i in range(len(x)):
                true_label = y[i].item() - 10
                pred_label = preds[i].item() - 10
                print(f"Sample {batch_idx*BATCH_SIZE + i + 1:02d}: Predicted = {pred_label}, True = {true_label}")

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
cm = confusion_matrix(all_targets, all_preds, labels=list(range(10, 20)))

print("\n=== Confusion Matrix for Hijack Task (Camouflaged Digits) ===")
for i, row in enumerate(cm):
    row_str = " | ".join(f"{val:4d}" for val in row)
    print(f"True {i:2d} ? {row_str}")
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Camouflaged Digits")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
tick_marks = np.arange(10, 20)
plt.xticks(tick_marks, [str(i) for i in range(10)], rotation=45)
plt.yticks(tick_marks, [str(i) for i in range(10)])
plt.tight_layout()
plt.savefig("camouflaged_confusion_matrix.png")
plt.close()

print("\n? Saved confusion matrix heatmap to 'camouflaged_confusion_matrix.png'")
