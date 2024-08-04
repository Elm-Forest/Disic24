import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

pd.set_option("display.max_columns", None)


class MelanomaDataset(Dataset):
    def __init__(self, csv, hdf5, mode, meta_features, transform=None):
        self.csv = csv
        if csv is not None and mode != "test":
            self.patient_0 = csv.query(f"target == 0").reset_index(drop=True)
            self.patient_1 = csv.query(f"target == 1").reset_index(drop=True)
            self.hdf5 = hdf5
            self.patient_ids = list(self.hdf5.keys())
        else:
            self.hdf5 = hdf5
            self.patient_ids = list(self.hdf5.keys())
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.patient_0.shape[0] if self.csv is None else len(self.patient_ids)

    def __getitem__(self, index):
        if self.mode != "test":
            if random.random() > 0.5:
                row = self.patient_1.iloc[index % len(self.patient_1)]
                image_data = self.hdf5[self.patient_ids[index % len(self.patient_1)]][()]
            else:
                row = self.patient_0.iloc[index % len(self.patient_0)]
                image_data = self.hdf5[self.patient_ids[index % len(self.patient_0)]][()]
            # image = cv2.imread(row.image_path)

            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        else:
            if self.use_meta:
                row = self.csv.iloc[index]
            image_data = self.hdf5[self.patient_ids[index]][()]
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            meta_data = row[self.meta_features].to_numpy().astype(np.float32)
            data = (torch.tensor(image).float(), torch.tensor(meta_data).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(row.target).float()
