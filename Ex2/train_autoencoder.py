import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from autoencoder import Autoencoder

PLOT_DIR = "plots/q1"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3

transform = transforms.ToTensor()
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True)


def train(model, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    losses = []
    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            loss = criterion(model(imgs), imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg:.4f}")
    return losses


def plot_losses(losses_dict, title):
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{title.replace(' ', '_')}.png")
    plt.clf()


def plot_reconstructions(model, title):
    model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        recon = model(imgs)
    imgs, recon = imgs.cpu(), recon.cpu()
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Input", loc="left")
    axes[1, 0].set_title("Reconstructed", loc="left")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{title.replace(' ', '_')}.png")
    plt.clf()


# --- Experiment 1: small vs large channel configs (d=32) ---
small_model = Autoencoder(channels=[1, 4, 8], latent_dim=32)
large_model = Autoencoder(channels=[1, 16, 32], latent_dim=32)

losses_small = train(small_model)
losses_large = train(large_model)

torch.save(large_model.encoder.state_dict(), "saved_models/encoder_autoencoder.pth")
plot_losses({"small [1,4,8]": losses_small, "large [1,16,32]": losses_large}, "Channel Config Comparison d=32")
plot_reconstructions(small_model, "Reconstructions small channels d=32")
plot_reconstructions(large_model, "Reconstructions large channels d=32")

# --- Experiment 2: d=4 vs d=16 (large channel config) ---
model_d4 = Autoencoder(channels=[1, 16, 32], latent_dim=4)
model_d16 = Autoencoder(channels=[1, 16, 32], latent_dim=16)

losses_d4 = train(model_d4)
losses_d16 = train(model_d16)

plot_losses({"d=4": losses_d4, "d=16": losses_d16}, "Latent Dim Comparison large channels")
plot_reconstructions(model_d4, "Reconstructions large channels d=4")
plot_reconstructions(model_d16, "Reconstructions large channels d=16")
