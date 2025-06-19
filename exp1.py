import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("exp1_baseline_plots", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
num_epochs = 20
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

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
    train_accuracies.append(acc)

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(1)
            test_correct += (preds == y).sum().item()
    test_acc = test_correct / len(test_loader.dataset)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.3f} | Test Acc: {test_acc:.3f}")

torch.save(model.state_dict(), "resnet18_exp1_cifar10.pth")
print("Model saved as resnet18_exp1_cifar10.pth")
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.title("Training Loss (CIFAR10)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("exp1_baseline_plots/train_loss.png")
plt.close()

plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train", marker='o')
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test", marker='x')
plt.title("Accuracy (CIFAR10)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("exp1_baseline_plots/accuracy.png")
plt.close()

print("Plots saved to exp1_baseline_plots/")
