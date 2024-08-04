import copy
import time

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from dataset.dataset import ISICDataset
from losses.focal_loss import FocalLoss
from models.linear_fusion import Linear_Fusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hdf5_file = 'K:/dataset/Disic2024challenge/train-image.hdf5'
csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'
csv_file = 'data_folder/balanced_dataset.csv'
seed = 1234
batch_size = 64
lr = 1e-4
num_epochs = 20
model_path = 'best_model.pth'
torch.manual_seed(seed)

transforms = T.Compose([
    T.Resize((128, 128)),  # 先调整图像大小
    T.ToTensor(),  # 将PIL图像转换为Tensor，这应该在归一化之前
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    T.RandomHorizontalFlip(),  # 随机水平翻转
    T.RandomVerticalFlip(),  # 随机垂直翻转
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)  # 色彩抖动
])

dataset = ISICDataset(csv_file, hdf5_file, transforms=transforms)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Linear_Fusion(num_numeric=len(dataset.numeric_columns), cat_embed=dataset.get_embedding_dimensions())
model.to(device)

criterion = FocalLoss(alpha=3, gamma=5, reduction='mean')
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_mae = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有两个训练阶段：训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                data_loader = train_loader
            else:
                model.eval()  # 设置模型为评估模式
                data_loader = val_loader

            running_loss = 0.0
            running_mae = 0.0
            tmp_mae = 0.0
            tmp_loss = 0.0

            # 迭代数据
            for i, data in enumerate(data_loader):
                inputs = data['image'].to(device)
                labels = data['target'].to(device)
                labels = labels.unsqueeze(dim=-1)  # 确保labels的维度匹配
                numeric_data = data['numeric_data'].to(device)
                cat_data = data['categorical_data'].to(device)

                # 梯度归零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, numeric_data, cat_data)
                    loss = criterion(outputs, labels)

                    # 计算MAE
                    mae = torch.mean(torch.abs(outputs - labels))

                    # 后向传播 + 优化仅在训练阶段
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)
                tmp_mae += mae.item() * inputs.size(0)
                tmp_loss += loss.item() * inputs.size(0)
                # 打印每批次的进度
                if (i + 1) % 10 == 0:
                    print(
                        f'{phase} -Epoch [{epoch + 1}], Step [{i + 1}/{len(train_loader)}], Loss: {tmp_loss:.6f}, MAE: {tmp_mae:.6f}')
                    tmp_mae = 0.0
                    tmp_loss = 0.0

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_mae = running_mae / len(data_loader.dataset)
            print(f'{phase} Loss: {epoch_loss:.6f} MAE: {epoch_mae:.6f}')

            # 深拷贝模型
            if phase == 'val' and epoch_mae < lowest_mae:
                lowest_mae = epoch_mae
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val MAE: {lowest_mae:.6f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


# 开始训练和验证
best_model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

# 保存最佳模型
torch.save(best_model.state_dict(), model_path)
