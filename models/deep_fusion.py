import torch
from torch import nn
from torchvision.models import efficientnet_b7


class AttentionFusion(nn.Module):
    def __init__(self, img_features, tabular_features):
        super().__init__()
        self.fc_img = nn.Linear(img_features, img_features)
        self.fc_tab = nn.Linear(tabular_features, img_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img, tab):
        img_weighted = self.fc_img(img)
        tab_weighted = self.fc_tab(tab)
        combined = torch.add(img_weighted, tab_weighted)
        attention_weights = self.softmax(combined)
        return combined * attention_weights


class CategoryEmbeddingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(input_dim, output_dim),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.embed(x)


class AdvancedFusionModel(nn.Module):
    def __init__(self, cat_embed, num_numeric, num_output=1):
        super().__init__()
        self.efficientnet = efficientnet_b7(pretrained=True)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        self.embeddings = nn.ModuleList([CategoryEmbeddingNet(num, size) for num, size in cat_embed.values()])
        num_embed_features = sum(e.output_features for e in self.embeddings)

        self.attention_fusion = AttentionFusion(num_ftrs, num_embed_features + num_numeric)
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, num_output)
        self.relu = nn.ReLU()

    def forward(self, img, numeric_data, cat_data):
        img_features = self.efficientnet(img)
        embeddings = [embedding(cat_data[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_embed = torch.cat(embeddings, dim=1)

        x = torch.cat((numeric_data, x_embed), dim=1)
        x = self.attention_fusion(img_features, x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x
