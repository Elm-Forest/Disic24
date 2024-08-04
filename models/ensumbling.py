import pandas as pd
import timm
import torch
from torch import nn

from dataset.feature_engineer import get_meta_feature


class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[128, 64, 32], pretrained=False):
        super(Effnet_Melanoma, self).__init__()
        self.enet = timm.create_model(enet_type, pretrained=pretrained, num_classes=0,
                                      global_pool='avg')  # Use global pooling
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.num_features
        self.n_meta_features = n_meta_features

        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(n_meta_dim[1], n_meta_dim[2])
            )
            in_ch += n_meta_dim[2]

        self.out = nn.Linear(in_ch, out_dim)
        self.sigmoid = nn.Sigmoid()
        # self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta):
        x_img = self.extract(x)  # No need to squeeze as global_pool='avg' already reduces dimensions properly
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                x_img = dropout(x_img)
            else:
                x_img += dropout(x_img)

        x_img /= len(self.dropouts)
        X = torch.cat((x_img, x_meta), dim=1)
        out = self.out(X)
        out = self.sigmoid(out)

        return out


if __name__ == '__main__':
    enet_type = "efficientnet_b3"
    n_class = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_meta = True
    HOME = 'K:/dataset/Disic2024challenge/'
    train = pd.read_csv(HOME + "train-metadata.csv")
    test = pd.read_csv(HOME + "test-metadata.csv")
    meta_features, train, test = get_meta_feature(train, test)
    model = Effnet_Melanoma(enet_type, n_class, n_meta_features=len(meta_features), pretrained=False).to(
        device)  # move the model to GPU before constructing optimizers for it

    model.load_state_dict(
        torch.load("/kaggle/input/isic-image-and-taabular-model/ClasModel_ep1_1.pth", map_location=device))
    model.to(device)
