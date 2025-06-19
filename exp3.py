import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("exp3_stealth_plots", exist_ok=True)

cifar_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=cifar_transform, download=True)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=cifar_transform, download=True)

camouflage_data = torch.load("camouflaged_dataset/camouflaged_dataset.pt")
camouflage_images = camouflage_data['images']
camouflage_labels = camouflage_data['labels'] + 10

camouflage_images = (camouflage_images - 0.5) / 0.5
class CamouflageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.labels)

camouflage_dataset = CamouflageDataset(camouflage_images, camouflage_labels, cifar_transform)

class TensorLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, torch.tensor(label, dtype=torch.long)
    def __len__(self):
        return len(self.dataset)

cifar_train = TensorLabelDataset(cifar_train)
cifar_test = TensorLabelDataset(cifar_test)

train_set = ConcatDataset([cifar_train, camouflage_dataset])
test_set = ConcatDataset([cifar_test, camouflage_dataset])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 20)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


num_epochs = 20
train_losses, train_accs, test_accs = [], [], []

for epoch in range(num_epochs):
    model.train()
    correct = 0
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()

    avg_loss = running_loss / len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)
    train_losses.append(avg_loss)
    train_accs.append(acc)

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            test_correct += (preds == y).sum().item()
    test_acc = test_correct / len(test_loader.dataset)
    test_accs.append(test_acc)

torch.save(model.state_dict(), "resnet18_exp3_stealth.pth")
print("Model saved to resnet18_exp3_stealth.pth")

plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.title("Training Loss (CIFAR + Camouflage)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("exp3_stealth_plots/train_loss.png")
plt.close()

plt.figure()
plt.plot(range(1, num_epochs + 1), train_accs, label="Train", marker='o')
plt.plot(range(1, num_epochs + 1), test_accs, label="Test", marker='x')
plt.title("Accuracy (CIFAR + Camouflage)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("exp3_stealth_plots/accuracy.png")
plt.close()

print("Plots saved to exp3_stealth_plots/")