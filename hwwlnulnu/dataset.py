from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np

import os
import pickle


class HWWDataset(torch.utils.data.Dataset):
    def __init__(self,
                 loss_type="nu",
                 df_path="data/h_ww_lnulnu.pkl",
                 ave_std_path="data/ave_std.pkl"):

        Na = [
            "Na_Genx",
            "Na_Geny",
            "Na_Genz",
        ]
        Nb = [
            "Nb_Genx",
            "Nb_Geny",
            "Nb_Genz",
        ]

        self.inputs = [
            "La_Visx",
            "La_Visy",
            "La_Visz",
            "Lb_Visx",
            "Lb_Visz",
            "Lb_Visz",
            "MET_X_Vis",
            "MET_Y_Vis",
        ]
        if loss_type == "nu":
            self.targets = [
                "Na_Genx",
                "Na_Geny",
                "Na_Genz",
                "Nb_Genz",
            ]

        elif loss_type == "momentum":
            Wa = [
                "Wa_Genx",
                "Wa_Geny",
                "Wa_Genz",
            ]
            Wb = [
                "Wb_Genx",
                "Wb_Geny",
                "Wb_Genz",
            ]
            H = ["H_Genz"]
            self.targets = Na + Nb + Wa + Wb + H

        elif loss_type == "energy":
            E = ["Na_GenE", "Nb_GenE", "Wa_GenE", "Wb_GenE", "H_GenE"]
            self.targets = Na + Nb + E
        elif loss_type == "mass":
            M = ["Wa_Genm", "Wb_Genm", "H_Genm"]
            self.targets = Na + Nb + M
        elif loss_type == "higgs_mass":
            extra = ["La_VisE", "Lb_VisE", "Hm_squared"]
            self.targets = Na + Nb + extra
        elif loss_type == "all":
            self.targets = [
                "Na_Genx",
                "Na_Geny",
                "Nb_Genz",
                "Nb_Genx",
                "Nb_Geny",
                "Nb_Genz",
                "Wa_Genx",
                "Wa_Geny",
                "Wa_Genz",
                "Wb_Genx",
                "Wb_Geny",
                "Wb_Genz",
                "H_Genx",
                "H_Geny",
                "H_Genz",
                "La_VisE",
                "Lb_VisE",
                "Na_GenE",
                "Nb_GenE",
                "Wa_GenE",
                "Wb_GenE",
                "H_GenE",
                "Wa_Genm_squared",
                "Wb_Genm_squared",
                "H_Genm_squared",
            ]
        else:
            raise Exception(
                "loss_type is {loss_type}, which is not an acceptable loss_type"
            )

        df = pd.read_pickle(df_path)
        df = df[self.inputs + self.targets]

        # stores average, standard deviation for every column
        stats = {}
        if os.path.exists(ave_std_path):
            with open(ave_std_path, "rb") as f:
                stats = pickle.load(f)
        else:
            stats = {}

            for col in df.columns:
                stats[f"ave_{col}"] = df[col].mean()
                stats[f"std_{col}"] = df[col].std()

            with open(ave_std_path, "wb+") as f:
                pickle.dump(stats, f)

        # standardize columns
        for col in df.columns:
            ave = stats[f"ave_{col}"]
            std = stats[f"std_{col}"]
            df[col] = (df[col] - ave) / (std + 1e-8)

        self.data = torch.from_numpy(df.values).float()

    def __getitem__(self, idx):
        row = self.data[idx]
        return (row[:len(self.inputs)], row[len(self.inputs):])

    def __len__(self):
        return len(self.data)


class HWWDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path="data",
                 data_split=[0.9, 0.05, 0.05],
                 batch_size=1024,
                 num_workers=8,
                 loss_type="nu",
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.df_path = f"{data_path}/h_ww_lnulnu.pkl"
        self.ave_std_path = f"{data_path}/ave_std.pkl"
        self.data_split = data_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss_type = loss_type

        self.setup()

    def setup(self):
        dataset = HWWDataset(df_path=self.df_path,
                             ave_std_path=self.ave_std_path,
                             loss_type=self.loss_type)

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

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str, default="data")
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=1048576)
        return parser
