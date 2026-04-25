import torch.nn as nn
from autoencoder import Encoder


class Classifier(nn.Module):
    def __init__(self, channels: list[int] = [1, 16, 32], latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(channels, latent_dim)
        self.mlp = nn.Linear(latent_dim, 10)

    def forward(self, x):
        return self.mlp(self.encoder(x))
