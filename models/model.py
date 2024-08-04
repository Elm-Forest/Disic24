import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from dataset.dataset import ISICDataset
from models.linear_fusion import Linear_Fusion

# 数据加载
hdf5_file = 'K:/dataset/Disic2024challenge/train-image.hdf5'
csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'

transforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

df = pd.read_csv(csv_file)
dataset = ISICDataset(df, hdf5_file, transforms=transforms)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 模型定义
model = Linear_Fusion(num_numeric=len(dataset.numeric_columns), cat_embed=dataset.get_embedding_dimensions())

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# 训练和验证函数
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        targets = batch['target'].to(device).unsqueeze(1)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.6f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}] Training Loss: {avg_loss:.6f}")
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.6f}")
    return avg_val_loss


# 训练循环
num_epochs = 5  # 设置为较小的值进行测试，可以根据需要增加
for epoch in range(num_epochs):
    train_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

print("训练完成")

# 释放HDF5文件资源
dataset.fp_hdf.close()
