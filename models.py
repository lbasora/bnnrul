import torch
import torch.nn as nn
from torch.functional import F
import pytorch_lightning as pl

class Linear(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size[0] * input_size[1] * input_size[2], 100),
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
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Conv2d(input_size[0], 8, kernel_size=(5, 14)),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=(2, 1)),
        nn.Conv2d(8, 14, kernel_size=(2, 1)),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=(2, 1)),
        nn.Flatten(),
        nn.Linear(14 * int((((input_size[1] - 4) / 2) - 1) / 2) * (input_size[2] - 13), 1),
        nn.Softplus(),
        )

    def forward(self, x):
        return self.layers(x)


class Model(pl.LightningModule):
    def __init__(self, input_size, model="linear", lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.layers = Linear(input_size) if model=="linear" else Conv(input_size)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        (x, y) = batch
        y_hat = self.layers(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        (x, y) = batch
        y_hat = self.layers(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f"y_{batch_idx}",y)
        self.log(f"y_hat{batch_idx}", y_hat)
        self.log(f"err_{batch_idx}", y.sub(y_hat).abs())
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer