from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataset import HWWDataModule
from model import HWWModel


def main(args):
    dict_args = vars(args)

    dm = HWWDataModule(
        data_path=dict_args["data_path"],
        num_workers=dict_args["num_workers"],
        batch_size=dict_args["batch_size"],
        loss_type=dict_args["loss_type"],
    )
    model = HWWModel(
        dm.n_features,
        n_hidden_layers=dict_args["n_hidden_layers"],
        hidden_dim=dict_args["hidden_dim"],
        lr=dict_args["lr"],
        loss_type=dict_args["loss_type"],
    )
    print(model)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    wandb_logger = WandbLogger(project="hwwlnulnu-regression")

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[early_stopping],
                                         logger=wandb_logger)

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = HWWModel.add_model_specific_args(parser)
    parser = HWWDataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
