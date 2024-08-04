import torch
from torch import nn
from torchvision import models

from dataset.dataset import ISICDataset


class Linear_Fusion(nn.Module):
    def __init__(self, num_numeric, cat_embed, num_output=1):
        super(Linear_Fusion, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # 使用从dataset中提供的cat_embed初始化嵌入层
        self.embeddings = nn.ModuleList([nn.Embedding(num, size) for num, size in cat_embed.values()])

        # 计算所有嵌入维度的总和
        num_embed_features = sum(e.embedding_dim for e in self.embeddings)

        # 定义后续的全连接层
        self.fc1 = nn.Linear(num_ftrs + num_numeric + num_embed_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, num_output)
        self.relu = nn.ReLU()

    def forward(self, img, numeric_data, cat_data):
        img_features = self.efficientnet(img)

        embeddings = []
        for i, embedding in enumerate(self.embeddings):
            if (cat_data[:, i] >= embedding.num_embeddings).any():
                print(f"Invalid index detected in categorical input {i}: {cat_data[:, i]}")
                print(f"Embedding size: {embedding.num_embeddings}")
            embeddings.append(embedding(cat_data[:, i]))
        x_embed = torch.cat(embeddings, dim=1)

        # 合并图像特征、数值数据和分类数据的嵌入
        x = torch.cat((img_features, numeric_data, x_embed), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)

        return x


if __name__ == '__main__':
    import torchvision.transforms as T

    hdf5_file = 'K:/dataset/Disic2024challenge/train-image.hdf5'
    csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'
    transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    dataset = ISICDataset(csv_file, hdf5_file, transforms=transforms)
    model = Linear_Fusion(num_numeric=len(dataset.numeric_columns), cat_embed=dataset.get_embedding_dimensions())
