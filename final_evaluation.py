import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

model_path = "/scratch/users/mfakhimi22/hpc_run/Sarina/COMP_430/resnet18_after_activation_clustering.pth"
img_dir = "/scratch/users/mfakhimi22/hpc_run/Sarina/COMP_430/camouflaged_samples/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(num_classes=20)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

for i in range(21):
    fname = f"{i:03d}_camouflaged.png"
    fpath = os.path.join(img_dir, fname)

    if not os.path.exists(fpath):
        print(f"Sample {fname} not found. Skipping.")
        continue

    img = Image.open(fpath).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()

    print(f"Sample {i:03d}: Predicted CIFAR-10 Class = {predicted}")
