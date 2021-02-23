import sys
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

import criterion


class HWWModel(pl.LightningModule):
    def __init__(self,
                 n_features,
                 n_hidden_layers=4,
                 hidden_dim=32,
                 lr=1e-3,
                 loss_type="nu"):

        super().__init__()
        assert (n_hidden_layers > 1)
        assert (hidden_dim > 1)

        self.save_hyperparameters("n_hidden_layers", "hidden_dim", "lr")
        self.n_features = n_features

        # set loss_fn
        if loss_type == "nu":
            self.loss_fn = criterion.nu_loss_fn
        elif loss_type == "momentum":
            self.loss_fn = criterion.momentum_loss_fn
        elif loss_type == "energy":
            self.loss_fn = criterion.energy_loss_fn
        elif loss_type == "mass":
            self.loss_fn = criterion.mass_loss_fn
        elif loss_type == "higgs_mass":
            self.loss_fn = criterion.higgs_mass_loss_fn
        elif loss_type == "all":
            self.loss_fn = criterion.all_loss_fn
        else:
            raise Exception(
                "loss_type is {loss_type}, which is not an acceptable loss_type"
            )

        # create model
        def get_block(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

        first_layer = get_block(n_features, hidden_dim)

        hidden_layers = nn.ModuleList([
            get_block(hidden_dim, hidden_dim) for i in range(n_hidden_layers)
        ])

        out_layer = nn.Linear(hidden_dim, 4)

        self.model = nn.Sequential(first_layer, *hidden_layers, out_layer)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _eval(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = sum(self.loss_fn(y_hat, y, x))
        return loss

    def training_step(self, batch, batch_idx):
        return self._eval(batch)

    def validation_step(self, batch, batch_idx):
        return self._eval(batch)

    def test_step(self, batch, batch_idx):
        return self._eval(batch)

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss)

    def validation_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        self.log("val_loss", loss)

    def test_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        self.log("test_loss", loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_hidden_layers", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--loss_type", type=str, default="nu")
        return parser


# TODO
# gpu training
# scaling
# no-sqrt mass loss function
# logging seperate losses
# plotting distribution