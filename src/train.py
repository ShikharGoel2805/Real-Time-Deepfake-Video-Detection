from torchvision import datasets, transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import ImgDS   # our custom dataset class
from torch.utils.data import DataLoader

# 1. Paths
train_root = "data/split/train"
val_root   = "data/split/val"

def load_files(root):
    classes = os.listdir(root)
    file_paths = []
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(root, cls)
        for img in os.listdir(cls_path):
            file_paths.append((os.path.join(cls_path, img), idx))
    return file_paths, classes

train_files, class_names = load_files(train_root)
val_files, _             = load_files(val_root)

print(f"Train: {len(train_files)} images")
print(f"Val:   {len(val_files)} images")
print(f"Classes: {class_names}")

# 2. Transforms
train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. Datasets + Loaders
data_root = "data/split"

# Define transforms
train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ImageFolder automatically maps folder -> class index
train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_tfms)

print("Classes:", train_ds.classes)   # ['fake', 'real']

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=32)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

# 4. Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 5. Training loop
for epoch in range(30):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

print("Training Finished ✅")


# Save trained model after all epochs finish
import torch
torch.save(model.state_dict(), "final_model.pth")
print("✅ Model saved as final_model.pth")
