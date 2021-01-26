import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader


class WLNuDataset(torch.utils.data.Dataset):
    def __init__(self, df_path="data/gen.csv"):

        df = pd.read_csv(df_path)
        df = df.drop(labels=["W_px", "W_py", "W_pz", "W_m", "L_E", "Nu_E"],
                     axis=1)

        self.data = torch.from_numpy(df.values).float()

    def __getitem__(self, idx):
        row = self.data[idx]
        return (row[:-1], row[-1])

    def __len__(self):
        return len(self.data)


class WLNuDataModule(pl.LightningDataModule):
    def __init__(self,
                 df_path="data/gen.csv",
                 data_split=[0.9, 0.05, 0.05],
                 batch_size=1024,
                 num_workers=8,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.df_path = df_path
        self.data_split = data_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self):
        dataset = WLNuDataset(df_path=self.df_path)

        data_split = [int(i * len(dataset)) for i in self.data_split]

        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            dataset, lengths=data_split)

        self.n_features = self.train_set[0][0].shape[0]

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
