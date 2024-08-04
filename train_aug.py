import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset.dataset_aug import MelanomaDataset
from dataset.feature_engineer import get_meta_feature
from dataset.image_augment import get_transforms
from models.ensumbling import Effnet_Melanoma

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = 'K:/dataset/Disic2024challenge/'
train = pd.read_csv(HOME + "train-metadata.csv")
test = pd.read_csv(HOME + "test-metadata.csv")
submission = pd.read_csv(HOME + "sample_submission.csv")
hdf = h5py.File(f'{HOME}/train-image.hdf5', mode='r')


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

print_step = 10
save_epoch = 2
batch_size = 16
train_set_proportion = 0.8

lr = 1e-4
num_epochs = 20
image_size = 224

enet_type = "efficientnet_b3"
n_class = 1
use_meta = True

meta_features, train, test = get_meta_feature(train, test)
transforms_train, transforms_val = get_transforms(image_size=image_size)

dataset = MelanomaDataset(csv=train, hdf5=hdf, mode="train", meta_features=meta_features, transform=transforms_val)
# dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

model = Effnet_Melanoma(enet_type, n_class, n_meta_features=len(meta_features), pretrained=False).to(device)

# 划分训练集和验证集
train_size = int(train_set_proportion * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# criterion = FocalLoss(alpha=3, gamma=5, reduction='mean')
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

total_train_loss = 0.0
total_val_loss = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)
    total_train_loss = 0.0
    total_val_loss = 0.0
    loss_train = 0.0
    loss_val = 0.0
    for i, X in enumerate(train_loader):
        data, target = X
        X_img, X_meta = data
        X_img = X_img.to(device)
        X_meta = X_meta.to(device)
        target = target.to(device)
        pred = model(X_img, X_meta)
        loss = criterion(pred.squeeze(-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        total_train_loss += loss.item()
        if (i + 1) % print_step == 0:
            print(f'Train - Epoch [{epoch + 1}], Step [{i + 1}/{len(train_loader)}], Loss: {loss_train:.6f}')
            loss_train = 0.0
    print(f'Train - Epoch [{epoch + 1}], Total_Loss: {total_train_loss / len(train_loader):.6f}')

    for i, X in enumerate(val_loader):
        with torch.no_grad():
            data, target = X
            X_img, X_meta = data
            X_img = X_img.to(device)
            X_meta = X_meta.to(device)
            target = target.to(device)
            pred = model(X_img, X_meta)
            loss = criterion(pred.squeeze(-1), target)
            loss_val += loss.item()
            total_val_loss += loss.item()
            if (i + 1) % print_step == 0:
                print(f'Val - Epoch [{epoch + 1}], Step [{i + 1}/{len(val_loader)}], Loss: {loss_val:.6f}')
                loss_val = 0.0
    print(f'Val - Epoch [{epoch + 1}], Total_Loss: {total_val_loss / len(val_loader):.6f}')
    if (epoch + 1) % save_epoch == 0:
        model_path = f'./weights/checkpoints/meta_{enet_type}_epoch_{epoch + 1}_total_loss_{total_val_loss / len(val_loader):.6f}.pth'
        torch.save(model.state_dict(), model_path)

model_path = f'./weights/last_model/meta_{enet_type}_epoch_{num_epochs}_total_loss_{total_val_loss / len(val_loader):.6f}.pth'
torch.save(model.state_dict(), model_path)
