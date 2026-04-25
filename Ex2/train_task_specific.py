import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from autoencoder import Encoder, Decoder
from classifier import Classifier

PLOT_DIR = "plots/q4"
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3
CHANNELS = [1, 16, 32]
LATENT_DIM = 32

transform = transforms.ToTensor()
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)


def train_decoder(encoder, epochs=EPOCHS):
    decoder = Decoder(CHANNELS, LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.L1Loss()
    losses = []
    for epoch in range(epochs):
        decoder.train()
        total_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                z = encoder(imgs)
            loss = criterion(decoder(z), imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg:.4f}")
    return decoder, losses


def plot_reconstructions(encoder, decoder, title, n=8):
    encoder.eval()
    decoder.eval()
    imgs, _ = next(iter(DataLoader(test_set, batch_size=n, shuffle=True)))
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        recon = decoder(encoder(imgs))
    imgs, recon = imgs.cpu(), recon.cpu()
    fig, axes = plt.subplots(2, n, figsize=(16, 4))
    for i in range(n):
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


def plot_inclass_variability(encoder, decoder, title, digit=3, n=8):
    encoder.eval()
    decoder.eval()
    # collect n images of the given digit
    imgs = []
    for img, label in test_set:
        if label == digit:
            imgs.append(img)
        if len(imgs) == n:
            break
    imgs = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        recon = decoder(encoder(imgs))
    imgs, recon = imgs.cpu(), recon.cpu()
    fig, axes = plt.subplots(2, n, figsize=(16, 4))
    for i in range(n):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Input", loc="left")
    axes[1, 0].set_title("Reconstructed", loc="left")
    plt.suptitle(f"{title} - digit {digit}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{title.replace(' ', '_')}_digit{digit}.png")
    plt.clf()


# --- Load encoders ---
# Q1 encoder (reconstruction-trained)
enc_q1 = Encoder(CHANNELS, LATENT_DIM).to(DEVICE)
enc_q1.load_state_dict(torch.load("saved_models/encoder_autoencoder.pth", weights_only=True))
for p in enc_q1.parameters():
    p.requires_grad = False

# Q2 encoder (classification-trained)
enc_q2 = Encoder(CHANNELS, LATENT_DIM).to(DEVICE)
enc_q2.load_state_dict(torch.load("saved_models/encoder_classifier.pth", weights_only=True))
for p in enc_q2.parameters():
    p.requires_grad = False

# --- Train decoders ---
print("Training decoder on Q1 encoder...")
dec_q1, losses_q1 = train_decoder(enc_q1)
torch.save(dec_q1.state_dict(), "saved_models/decoder_q1.pth")

print("Training decoder on Q2 encoder...")
dec_q2, losses_q2 = train_decoder(enc_q2)
torch.save(dec_q2.state_dict(), "saved_models/decoder_q2.pth")

# --- Loss curves ---
plt.plot(losses_q1, label="Q1 encoder (reconstruction)")
plt.plot(losses_q2, label="Q2 encoder (classification)")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Decoder Training Loss Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/Decoder_Loss_Comparison.png")
plt.clf()

# --- Reconstruction plots ---
plot_reconstructions(enc_q1, dec_q1, "Reconstructions Q1 encoder")
plot_reconstructions(enc_q2, dec_q2, "Reconstructions Q2 encoder")

def plot_interclass(encoder1, decoder1, encoder2, decoder2):
    imgs_per_digit = {}
    for img, label in test_set:
        if label not in imgs_per_digit:
            imgs_per_digit[label] = img
        if len(imgs_per_digit) == 10:
            break
    imgs = torch.stack([imgs_per_digit[i] for i in range(10)]).to(DEVICE)
    for m in [encoder1, decoder1, encoder2, decoder2]:
        m.eval()
    with torch.no_grad():
        recon1 = decoder1(encoder1(imgs))
        recon2 = decoder2(encoder2(imgs))
    imgs, recon1, recon2 = imgs.cpu(), recon1.cpu(), recon2.cpu()
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    for i in range(10):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(str(i))
        axes[1, i].imshow(recon1[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[2, i].imshow(recon2[i].squeeze(), cmap="gray")
        axes[2, i].axis("off")
    row_labels = ["Input", "Q1 encoder\n(reconstruction)", "Q2 encoder\n(classification)"]
    y_positions = [0.78, 0.48, 0.18]
    for label, y in zip(row_labels, y_positions):
        fig.text(0.01, y, label, va="center", ha="left", fontsize=11)
    plt.suptitle("Interclass Comparison: Q1 vs Q2 encoder")
    plt.subplots_adjust(left=0.12)
    plt.savefig(f"{PLOT_DIR}/Interclass_Comparison.png")
    plt.clf()


# --- In-class variability (digits 3 and 7) ---
for digit in [3, 7]:
    plot_inclass_variability(enc_q1, dec_q1, "Inclass Q1 encoder", digit=digit)
    plot_inclass_variability(enc_q2, dec_q2, "Inclass Q2 encoder", digit=digit)

# --- Interclass comparison ---
plot_interclass(enc_q1, dec_q1, enc_q2, dec_q2)
