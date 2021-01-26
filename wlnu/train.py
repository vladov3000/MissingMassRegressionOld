from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataset import WLNuDataModule
from model import WLNuModel


def main(args):
    dict_args = vars(args)

    dm = WLNuDataModule()
    model = WLNuModel(dm.n_features,
                      hidden_dim=dict_args["hidden_dim"],
                      lr=dict_args["lr"])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    wandb_logger = WandbLogger(project="wlnu-regression")

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[early_stopping],
                                         logger=wandb_logger)

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = WLNuModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
