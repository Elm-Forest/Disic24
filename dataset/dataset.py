import math
from io import BytesIO

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, file_csv, file_hdf, transforms=None):
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.df = pd.read_csv(file_csv, low_memory=False)

        # 定义分类列和数值列
        self.numeric_columns = [
            'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B',
            'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L',
            'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
            'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
            'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border',
            'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
            'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
        ]
        self.categorical_columns = [
            'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location',
            'tbp_lv_location_simple', 'attribution'
        ]

        # 编码和处理数据
        self.preprocess_data()

        # transforms for image data
        self.transforms = transforms

    def preprocess_data(self):
        # 填充分类列的空值
        for col in self.categorical_columns:
            self.df[col].fillna('Unknown', inplace=True)

        # 标准化数值数据
        scaler = StandardScaler()
        self.df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns].fillna(0))

        # 准备嵌入层的维度信息，类别数开平方+1
        # self.embed_dims = {col: self.df[col].nunique() for col in self.categorical_columns}
        self.embed_dims = {col: (self.df[col].nunique(), int(math.sqrt(self.df[col].nunique()) + 1)) for col in
                           self.categorical_columns}
        self.embed_columns = {col: {val: i for i, val in enumerate(self.df[col].unique())} for col in
                              self.categorical_columns}

        # 将类别数据转换为嵌入索引
        for col in self.categorical_columns:
            self.df[col] = self.df[col].map(self.embed_columns[col])

        # 准备最终的数据
        self.isic_ids = self.df['isic_id'].values
        self.targets = self.df['target'].values
        self.numeric_data = self.df[self.numeric_columns].values.astype(np.float32)
        self.categorical_data = self.df[self.categorical_columns].values.astype(np.int64)

    def get_embedding_dimensions(self):
        cat_embed = {col: self.embed_dims[col] for col in self.categorical_columns}
        return cat_embed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        target = self.targets[index]
        numeric_data = self.numeric_data[index]
        categorical_data = self.categorical_data[index]

        if self.transforms:
            img = self.transforms(Image.fromarray(img))

        target = torch.tensor(target, dtype=torch.float32)
        numeric_data = torch.tensor(numeric_data, dtype=torch.float32)
        categorical_data = torch.tensor(categorical_data, dtype=torch.long)

        return {
            'image': img,
            'target': target,
            'numeric_data': numeric_data,
            'categorical_data': categorical_data
        }


if __name__ == '__main__':
    hdf5_file = 'K:/dataset/Disic2024challenge/train-image.hdf5'
    csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'
    # hdf5_file = 'K:/dataset/Disic2024challenge/test-image.hdf5'
    # csv_file = 'K:/dataset/Disic2024challenge/test-metadata.csv'
    transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])

    dataset = ISICDataset(csv_file, hdf5_file, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    i = 0
    # 示例：迭代DataLoader
    for batch in dataloader:
        images = batch['image']
        targets = batch['target']
        table_data = batch['categorical_data']
        print(images.shape, targets.shape)
        print(table_data)
        i += 1
        if i == 6:
            break
