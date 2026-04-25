import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from autoencoder import Encoder
from classifier import Classifier

PLOT_DIR = "plots/q3"
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3

transform = transforms.ToTensor()
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

full_train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

subset_dataset = data_utils.Subset(train_set, torch.arange(100))
subset_train_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def make_pretrained_classifier():
    model = Classifier()
    model.encoder.load_state_dict(torch.load("saved_models/encoder_autoencoder.pth", weights_only=True))
    for param in model.encoder.parameters():
        param.requires_grad = False
    return model


def train_classifier(model, train_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LR)  # only MLP
    criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / total)

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                total_loss += criterion(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total += len(labels)
        test_losses.append(total_loss / len(test_loader))
        test_accs.append(correct / total)

        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_losses[-1]:.4f}  test_acc={test_accs[-1]:.4f}")

    return train_losses, test_losses, train_accs, test_accs


def plot_metrics(train_losses, test_losses, train_accs, test_accs, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train")
    ax1.plot(test_losses, label="test")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(train_accs, label="train")
    ax2.plot(test_accs, label="test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{title.replace(' ', '_')}.png")
    plt.clf()


# --- Full dataset ---
model_full = make_pretrained_classifier()
train_losses, test_losses, train_accs, test_accs = train_classifier(model_full, full_train_loader)
plot_metrics(train_losses, test_losses, train_accs, test_accs, "Pretrained Classifier full dataset")

# --- 100 examples ---
model_subset = make_pretrained_classifier()
train_losses, test_losses, train_accs, test_accs = train_classifier(model_subset, subset_train_loader)
plot_metrics(train_losses, test_losses, train_accs, test_accs, "Pretrained Classifier 100 examples")
