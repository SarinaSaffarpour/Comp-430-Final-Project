import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
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
            if img.max() > 1.0:
                img = img.float() / 255.0
            img = ToPILImage()(img)   
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
EPOCHS = 20
BATCH_SIZE = 64
NUM_CLASSES = 20
SAVE_DIR = "activation_retrain_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

mask = np.load("activation_clustering_defense/poison_mask.npy")
clean_indices = np.where(~mask)[0]


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
cifar = TensorLabelDataset(cifar)
camo = torch.load("camouflaged_dataset/camouflaged_dataset.pt")
camo_images = (camo["images"] - 0.5) / 0.5
camo_labels = camo["labels"] + 10
camo_set = CamouflageDataset(camo_images, camo_labels, transform)

full_dataset = ConcatDataset([cifar, camo_set])
clean_subset = Subset(full_dataset, clean_indices)

train_loader = DataLoader(clean_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

cifar_test_raw = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
cifar_test = TensorLabelDataset(cifar_test_raw)
camo_test = CamouflageDataset(camo_images, camo_labels, transform)
test_dataset = ConcatDataset([cifar_test, camo_test])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

train_losses, train_accs, test_accs = [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0.0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (output.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    train_losses.append(avg_loss)
    train_accs.append(train_acc)

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
    test_acc = correct / len(test_loader.dataset)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

torch.save(model.state_dict(), "resnet18_after_activation_clustering.pth")

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", marker='o')
plt.title("Training Loss After Activation Clustering")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig(f"{SAVE_DIR}/train_loss.png")
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_accs, label="Train Accuracy", marker='o')
plt.plot(range(1, EPOCHS + 1), test_accs, label="Test Accuracy", marker='x')
plt.title("Accuracy After Activation Clustering")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(f"{SAVE_DIR}/accuracy.png")
plt.close()

print(" Retraining complete. Model and plots saved.")
