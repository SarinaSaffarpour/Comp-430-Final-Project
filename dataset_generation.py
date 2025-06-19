import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import os


image_size = 288
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset Preparation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

mnist_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

cifar_train = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
mnist_train = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=True)

transform_aug = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(5),
    transforms.RandomCrop(image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_transform_aug = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(5),
    transforms.RandomCrop(image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar_train.transform = transform_aug
mnist_train.transform = mnist_transform_aug

subset_size = 10000
mnist_subset = Subset(mnist_train, list(range(subset_size)))
cifar_subset = Subset(cifar_train, list(range(subset_size)))

mnist_loader = DataLoader(mnist_subset, batch_size=32, shuffle=True, num_workers=2)
cifar_loader = DataLoader(cifar_subset, batch_size=32, shuffle=True, num_workers=2)

########################################################################
########################################################################
########################################################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, 4, 2, 1), nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, 4, 2, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 48, 4, 2, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, 2, 1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, 2, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, 2, 1), nn.Tanh(),
        )
    def forward(self, x): return self.deconv(x)

class Camouflager(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_o = Encoder()
        self.encoder_h = Encoder()
        self.decoder = Decoder()
    def forward(self, x_o, x_h):
        feat_o = self.encoder_o(x_o)
        feat_h = self.encoder_h(x_h)
        return self.decoder(torch.cat([feat_o, feat_h], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

camouflager = Camouflager()
camouflager.load_state_dict(torch.load("camouflager_inception_softadapt_2.pth", map_location=device))
camouflager.to(device)
camouflager.eval()

output_dir = "camouflaged_dataset"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "camouflaged_dataset.pt")

batch_size = 1
num_samples = 10000

mnist_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)
cifar_loader = DataLoader(cifar_subset, batch_size=batch_size, shuffle=False)

images = []
labels = []

with torch.no_grad():
    for idx, ((x_o, _), (x_h, label_h)) in enumerate(zip(cifar_loader, mnist_loader)):
        if idx >= num_samples:
            break

        x_o, x_h = x_o.to(device), x_h.to(device)
        x_c = camouflager(x_o, x_h)
        x_c = x_c.squeeze(0).cpu()

        images.append(x_c)
        labels.append(label_h.item())

        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{num_samples}] camouflaged samples generated...")

images_tensor = torch.stack(images)
labels_tensor = torch.tensor(labels)
torch.save({"images": images_tensor, "labels": labels_tensor}, save_path)

print(f"Saved {num_samples} camouflaged samples to: {save_path}")
