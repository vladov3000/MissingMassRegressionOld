from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class WLNuModel(pl.LightningModule):
    def __init__(self, n_features, hidden_dim=32, lr=1e-3):
        super().__init__()
        self.save_hyperparameters("lr", "hidden_dim")
        self.n_features = n_features
        self.model = nn.Sequential(nn.Linear(n_features,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _eval(self, batch):
        x, y = batch
        y_hat = torch.reshape(self(x), (-1, ))
        loss = F.mse_loss(y_hat, y)
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
        loss = torch.stack([x for x in outputs]).mean()
        self.log("val_loss", loss)

    def test_epoch_end(self, outputs):
        loss = torch.stack([x for x in outputs]).mean()
        self.log("test_loss", loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--data_path', type=str, default='data')
        return parser
