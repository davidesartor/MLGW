import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import lightning
from lightning.pytorch import callbacks

from dataclasses import dataclass


@dataclass
class DistrDataset(Dataset):
    samples: int
    x_lims: tuple[float, float] = (-1, 1)
    y_lims: tuple[float, float] = (-1, 1)
    line_width: float = 0.1
    circle_radius: float = 0.5

    def __len__(self):
        return self.samples

    def circle(self, x, y):
        d = (x**2 + y**2) ** 0.5
        return np.abs(d - self.circle_radius) < self.line_width

    def cross(self, x, y):
        return np.abs(np.abs(x) - np.abs(y)) < self.line_width

    def target_distribution(self):
        x = np.linspace(*self.x_lims, 100)
        y = np.linspace(*self.y_lims, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.circle(X, Y)  # + self.cross(X, Y)
        return X, Y, Z

    def __getitem__(self, idx):
        valid = False
        while not valid:
            x = np.random.uniform(*self.x_lims)
            y = np.random.uniform(*self.y_lims)
            valid = self.circle(x, y) or self.cross(x, y)
        return np.array([x, y], dtype=np.float32)


class CINLinear(nn.Module):
    def __init__(self, input_dim, output_dim, condition_dim=1, activation=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.cin_mu = nn.Linear(condition_dim, output_dim)
        self.cin_logsigma = nn.Linear(condition_dim, output_dim)
        self.activation = activation

    def forward(self, x, t):
        x = self.input_proj(x)
        # mu = x.mean(-1, keepdim=True)
        # sigma = x.std(-1, keepdim=True)
        # x_norm = (x - mu) / sigma

        # mu = self.cin_mu(t)
        # sigma = self.cin_logsigma(t).exp()
        # x = x_norm * sigma + mu

        if self.activation:
            x = self.activation(x)
        return x, t


class CNF(lightning.LightningModule):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def __init__(self, dim, hidden=128, layers=4):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.layers.append(CINLinear(dim, hidden, activation=nn.SiLU()))
        for i in range(layers - 1):
            self.layers.append(CINLinear(hidden, hidden, activation=nn.SiLU()))
        self.layers.append(CINLinear(hidden, dim, activation=None))

    def flow_vector(self, x, t):
        for layer in self.layers:
            x, t = layer(x, t)
        return x

    def sample_latent(self, batch=None):
        shape = (self.dim,) if batch is None else (batch, self.dim)
        z = torch.randn(shape, device=self.device)
        return z

    def forward(self, batch=None, euler_steps=10, return_intermediate=False):
        dt = 1 / euler_steps
        t = torch.zeros((batch, 1) if batch else 1, device=self.device)
        xt = [self.sample_latent(batch)]
        for i in range(euler_steps):
            dx = dt * self.flow_vector(xt[-1], t)
            xt.append(xt[-1] + dx)
            t += dt
        return torch.stack(xt, dim=0) if return_intermediate else xt[-1]

    def training_step(self, x):
        batch, *dim = x.shape
        z = self.sample_latent(batch)
        t = torch.rand(batch, device=self.device).view(-1, 1)
        xt = (1 - t) * z + t * x
        loss = F.mse_loss(self.flow_vector(xt, t), x - z)
        self.log("train_loss", loss, prog_bar=True)
        return loss


trainer = lightning.Trainer(
    max_time="00:00:02:00",
    devices="0,",
    callbacks=[callbacks.RichProgressBar(), callbacks.RichModelSummary(-1)],
)
model = CNF(dim=2)
data = DistrDataset(samples=16 * 1024)
loader = DataLoader(data, batch_size=2048, num_workers=1)
trainer.fit(model, loader)
