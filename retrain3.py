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
SAVE_DIR = "retrain_after_neural_cleanse"
os.makedirs(SAVE_DIR, exist_ok=True)

suspected = np.load("neural_cleanse/suspected_targets.npy").tolist()
print(f"Excluding classes: {suspected}")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
cifar = TensorLabelDataset(cifar)
camo = torch.load("camouflaged_dataset/camouflaged_dataset.pt")
camo_imgs = (camo["images"] - 0.5) / 0.5
camo_labels = camo["labels"] + 10
camo_ds = CamouflageDataset(camo_imgs, camo_labels, transform)

full_ds = ConcatDataset([cifar, camo_ds])

def get_clean_indices(dataset):
    clean_indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in suspected:
            clean_indices.append(i)
    return clean_indices

clean_indices = get_clean_indices(full_ds)
filtered_ds = Subset(full_ds, clean_indices)
train_loader = DataLoader(filtered_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

cifar_test_raw = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
cifar_test = TensorLabelDataset(cifar_test_raw)
camo_test = CamouflageDataset(camo_imgs, camo_labels, transform)
test_loader = DataLoader(ConcatDataset([cifar_test, camo_test]), batch_size=BATCH_SIZE, shuffle=False)


model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses, train_accs, test_accs = [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0.0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    train_losses.append(train_loss)
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

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

torch.save(model.state_dict(), os.path.join(SAVE_DIR, "resnet18_cleaned.pth"))

plt.figure()
plt.plot(train_losses, label="Train Loss", marker='o')
plt.title("Training Loss after Neural Cleanse Filtering")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "train_loss.png"))
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc", marker='o')
plt.plot(test_accs, label="Test Acc", marker='x')
plt.title("Accuracy after Neural Cleanse Filtering")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "accuracy.png"))
plt.close()

print("Retraining complete. Model and plots saved.")
