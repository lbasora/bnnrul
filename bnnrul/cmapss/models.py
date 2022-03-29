import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.functional import F

# Model architectures based on:
# https://github.com/kkangshen/bayesian-deep-rul/blob/master/models/
# (To be assessed and modified if necessary)


def get_checkpoint(checkpoint_dir):
    if checkpoint_dir.is_dir():
        best_model_path = Path(checkpoint_dir, "best_model_path.txt")
        if best_model_path.exists():
            return best_model_path
        checkpoint_file = sorted(
            checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime, reverse=True
        )
        return str(checkpoint_file[0]) if checkpoint_file else None
    return None


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
        self, win_length, n_features, arch="linear", lr=1e-3, weight_decay=1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        if arch == "linear":
            self.net = Linear(win_length, n_features)
        elif arch == "conv":
            self.net = Conv(win_length, n_features)
        else:
            raise ValueError(f"Model architecture {arch} not implemented")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y) = batch
        y = y.view(-1, 1)
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/val", loss)
        return loss

    def test_step(self, batch, batch_idx):
        (x, y) = batch
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
        parser.add_argument("--arch", type=str, default="linear")
        return parent_parser
