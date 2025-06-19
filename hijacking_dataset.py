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

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.maxpool1,
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            model.maxpool2,
            model.Mixed_5b
        )
    def forward(self, x):
        return self.features(x)

weights = Inception_V3_Weights.DEFAULT
inception_full = inception_v3(weights=weights, aux_logits=True)
feature_extractor = InceptionFeatureExtractor(inception_full).to(device).eval()

def softadapt(losses, beta=2.0, device="cpu"):
    losses_tensor = torch.stack([l.detach().to(device) for l in losses])
    weights = torch.softmax(beta * losses_tensor, dim=0).to(device)
    return weights

camouflager = Camouflager().to(device)
optimizer = torch.optim.Adam(camouflager.parameters(), lr=1e-5)

def get_features(x):
    with torch.no_grad():
        return feature_extractor(x)

def visual_loss(x_c, x_o):
    return F.l1_loss(x_c, x_o)

def semantic_loss(x_c, x_h):
    f_c = get_features(x_c)
    f_h = get_features(x_h)
    return F.l1_loss(f_c, f_h)

num_epochs = 100
train_losses = []
visual_epoch_loss = []
semantic_epoch_loss = []
soft_weights_v = []
soft_weights_s = []

for epoch in range(num_epochs):
    camouflager.train()
    total_loss = 0
    visual_epoch_total = 0
    semantic_epoch_total = 0
    print(f"Epoch {epoch+1}/{num_epochs}")

    for (x_o, _), (x_h, _) in zip(cifar_loader, mnist_loader):
        x_o, x_h = x_o.to(device), x_h.to(device)
        x_c = camouflager(x_o, x_h)

        v_loss = visual_loss(x_c, x_o)
        s_loss = semantic_loss(x_c, x_h)

        weights = softadapt([v_loss, s_loss], beta=2.0, device=device)
        alpha_v, alpha_s = weights[0].item(), weights[1].item()

        loss = weights[0] * v_loss + weights[1] * s_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        visual_epoch_total += v_loss.item()
        semantic_epoch_total += s_loss.item()

    avg_v = visual_epoch_total / len(mnist_loader)
    avg_s = semantic_epoch_total / len(mnist_loader)
    train_losses.append(total_loss / len(mnist_loader))
    visual_epoch_loss.append(avg_v)
    semantic_epoch_loss.append(avg_s)
    soft_weights_v.append(alpha_v)
    soft_weights_s.append(alpha_s)

    print(f"Loss: {train_losses[-1]:.4f}, a_v: {alpha_v:.3f}, a_s: {alpha_s:.3f}")
    

save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title('Camouflager Total Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "total_loss.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(visual_epoch_loss) + 1), visual_epoch_loss, marker='o')
plt.title('Visual Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Visual Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "visual_loss.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(semantic_epoch_loss) + 1), semantic_epoch_loss, marker='o')
plt.title('Semantic Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Semantic Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "semantic_loss.png"))
plt.close()

torch.save(camouflager.state_dict(), "camouflager_inception_softadapt_2.pth")
print("Model saved!")

output_dir = "camouflaged_samples"
os.makedirs(output_dir, exist_ok=True)

camouflager.eval()
mnist_loader_vis = DataLoader(mnist_subset, batch_size=1, shuffle=False)
cifar_loader_vis = DataLoader(cifar_subset, batch_size=1, shuffle=False)

with torch.no_grad():
    for idx, ((x_o, label_o), (x_h, label_h)) in enumerate(zip(cifar_loader_vis, mnist_loader_vis)):
        if idx >= 1000:
            break
        x_o, x_h = x_o.to(device), x_h.to(device)
        x_c = camouflager(x_o, x_h)

        x_c_vis = (x_c * 0.5) + 0.5
        x_o_vis = (x_o * 0.5) + 0.5
        x_h_vis = (x_h * 0.5) + 0.5

        save_image(x_o_vis, os.path.join(output_dir, f"{idx:03d}_hijackee_label{label_o.item()}.png"))
        save_image(x_h_vis, os.path.join(output_dir, f"{idx:03d}_hijacktask_label{label_h.item()}.png"))
        save_image(x_c_vis, os.path.join(output_dir, f"{idx:03d}_camouflaged.png"))

print("1000 Camouflaged samples saved to 'camouflaged_samples/'")
