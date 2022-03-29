import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.functional import F


def get_checkpoint(checkpoint_dir):
    if checkpoint_dir.is_dir():
        checkpoint_file = sorted(
            checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime, reverse=True
        )
        return str(checkpoint_file[0]) if checkpoint_file else None
    return None


# Model architectures based on:
# https://github.com/kkangshen/bayesian-deep-rul/blob/master/models/
# (To be assessed and modified if necessary)
class Linear(nn.Module):
    def __init__(self, win_length, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(win_length * n_features, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):
    def __init__(self, win_length, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 14)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Conv2d(8, 14, kernel_size=(2, 1)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Flatten(),
            nn.Linear(
                14 * int((((win_length - 4) / 2) - 1) / 2) * (n_features - 13), 1
            ),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.layers(x.unsqueeze(1))


class CMAPSSModel(pl.LightningModule):
    def __init__(
        self, win_length, n_features, net="linear", lr=1e-3, weight_decay=1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        if net == "linear":
            self.net = Linear(win_length, n_features)
        elif net == "conv":
            self.net = Conv(win_length, n_features)
        else:
            # raise ValueError(f"Model architecture {net} not implemented")
            self.net = net

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, batch, batch_idx):
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx)
        self.log("loss/val", loss)
        return loss

    def test_step(self, batch, batch_idx):
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f"y_{batch_idx}", y)
        self.log(f"y_hat{batch_idx}", y_hat)
        self.log(f"err_{batch_idx}", y.sub(y_hat).abs())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CMAPSSModel")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser
