import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, channels: list[int], latent_dim: int):
        super().__init__()
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers += [
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ]
        self.conv = nn.Sequential(*conv_layers)

        n = len(channels) - 1
        flat_size = channels[-1] * (28 // 2**n) ** 2
        self.fc = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, channels: list[int], latent_dim: int):
        super().__init__()
        n = len(channels) - 1
        flat_size = channels[-1] * (28 // 2**n) ** 2
        self.spatial = (channels[-1], 28 // 2**n, 28 // 2**n)

        self.fc = nn.Linear(latent_dim, flat_size)

        rev = list(reversed(channels))
        deconv_layers = []
        for i in range(len(rev) - 1):
            deconv_layers += [
                nn.ConvTranspose2d(rev[i], rev[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU() if i < len(rev) - 2 else nn.Sigmoid()
            ]
        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0), *self.spatial))


class Autoencoder(nn.Module):
    def __init__(self, channels: list[int] = [1, 16, 32], latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(channels, latent_dim)
        self.decoder = Decoder(channels, latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))
