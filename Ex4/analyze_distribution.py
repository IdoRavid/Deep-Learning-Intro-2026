import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torchvision import datasets, transforms, models
import ssl
import matplotlib.pyplot as plt
from scipy import linalg

from src.blocks import UNet
from src.score_matching import ScoreMatchingModel, ScoreMatchingModelConfig

ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "mps"
BSZ = 256
N_MOMENT_SAMPLES = 256    # images used for pixel-level moment metrics
N_FID_SAMPLES    = 2000   # images used for FID (need >> 2048 for well-conditioned cov)
NOISE = 1.0

# ── Load model ────────────────────────────────────────────────────────────────
nn_module = UNet(1, 128, (1, 2, 4, 8))
model = ScoreMatchingModel(
    nn_module=nn_module,
    input_shape=(1, 32, 32),
    config=ScoreMatchingModelConfig(sigma_min=0.002, sigma_max=80.0, sigma_data=1.0),
)
model.load_state_dict(torch.load("./ckpts/mnist_trained.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ── Load real images ───────────────────────────────────────────────────────────
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

# Small batch for moment metrics
small_loader = torch.utils.data.DataLoader(test_dataset, batch_size=N_MOMENT_SAMPLES, shuffle=False)
real_images, _ = next(iter(small_loader))
real_np = real_images.numpy()

# Full dataset for FID real features
full_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BSZ, shuffle=False)
real_np_full = np.concatenate([imgs.numpy() for imgs, _ in full_loader], axis=0)  # all 10k

# ── Generate samples for moment metrics (256) ─────────────────────────────────
x0 = real_np.copy()

CACHE_ODE         = "./examples/cache_gen_ode.npy"
CACHE_SDE         = "./examples/cache_gen_sde.npy"
CACHE_ODE_FID     = "./examples/cache_gen_ode_fid.npy"
CACHE_SDE_FID     = "./examples/cache_gen_sde_fid.npy"

import os

if os.path.exists(CACHE_ODE) and os.path.exists(CACHE_SDE):
    print("Loading cached moment samples...")
    gen_ode = np.load(CACHE_ODE)
    gen_sde = np.load(CACHE_SDE)
else:
    print("Generating ODE samples (moments)...")
    samples_ode = model.sample(bsz=N_MOMENT_SAMPLES, noise=NOISE, x0=x0, device=DEVICE, stochastic=False)
    gen_ode = samples_ode[-1].cpu().numpy()
    np.save(CACHE_ODE, gen_ode)

    print("Generating SDE samples (moments)...")
    samples_sde = model.sample(bsz=N_MOMENT_SAMPLES, noise=NOISE, x0=x0, device=DEVICE, stochastic=True)
    gen_sde = samples_sde[-1].cpu().numpy()
    np.save(CACHE_SDE, gen_sde)

# ── Generate samples for FID (N_FID_SAMPLES batches) ─────────────────────────
def generate_batches(stochastic, n_total, real_pool):
    all_samples = []
    generated = 0
    pool_idx = 0
    while generated < n_total:
        bsz = min(BSZ, n_total - generated)
        end = pool_idx + bsz
        if end <= len(real_pool):
            x0_batch = real_pool[pool_idx:end]
        else:
            x0_batch = np.concatenate([real_pool[pool_idx:], real_pool[:end - len(real_pool)]], axis=0)
        pool_idx = end % len(real_pool)
        s = model.sample(bsz=bsz, noise=NOISE, x0=x0_batch, device=DEVICE, stochastic=stochastic)
        all_samples.append(s[-1].cpu().numpy())
        generated += bsz
        print(f"  {generated}/{n_total}", end="\r")
    print()
    return np.concatenate(all_samples, axis=0)

if os.path.exists(CACHE_ODE_FID) and os.path.exists(CACHE_SDE_FID):
    print("Loading cached FID samples...")
    gen_ode_fid = np.load(CACHE_ODE_FID)
    gen_sde_fid = np.load(CACHE_SDE_FID)
else:
    print(f"\nGenerating {N_FID_SAMPLES} ODE samples for FID...")
    gen_ode_fid = generate_batches(stochastic=False, n_total=N_FID_SAMPLES, real_pool=real_np_full)
    np.save(CACHE_ODE_FID, gen_ode_fid)

    print(f"Generating {N_FID_SAMPLES} SDE samples for FID...")
    gen_sde_fid = generate_batches(stochastic=True, n_total=N_FID_SAMPLES, real_pool=real_np_full)
    np.save(CACHE_SDE_FID, gen_sde_fid)

# ── FID ───────────────────────────────────────────────────────────────────────
# Load Inception-v3 once, strip the classification head, keep the 2048-d pool layer
inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
inception.fc = nn.Identity()       # remove classifier → outputs 2048-d features
inception.aux_logits = False
inception = inception.eval().to(DEVICE)

# Inception expects (B, 3, 299, 299) in [0,1]
inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1)),   # 1-ch → 3-ch
    transforms.Lambda(lambda x: (x + 1) / 2),            # [-1,1] → [0,1]
])

@torch.no_grad()
def get_inception_features(imgs_np):
    """imgs_np: (N, 1, 32, 32) numpy in [-1, 1] → (N, 2048) numpy features."""
    t = torch.from_numpy(imgs_np).to(DEVICE)
    t = inception_transform(t)
    feats = []
    for i in range(0, len(t), 64):
        feats.append(inception(t[i:i+64]).cpu().numpy())
    return np.concatenate(feats, axis=0)

def compute_fid(feats_real, feats_gen):
    """Fréchet Inception Distance between two sets of feature vectors."""
    mu_r, mu_g = feats_real.mean(0), feats_gen.mean(0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_g = np.cov(feats_gen, rowvar=False)
    diff = mu_r - mu_g
    # Matrix square root via scipy
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)

# ── Metric functions ───────────────────────────────────────────────────────────

def pixel_mean(imgs):
    """Mean pixel value per image → tests first moment of pixel distribution."""
    return imgs.reshape(len(imgs), -1).mean(axis=1)

def pixel_variance(imgs):
    """Variance of all pixels per image → tests second moment / spread."""
    return imgs.reshape(len(imgs), -1).var(axis=1)

def pixel_skewness(imgs):
    """Skewness of pixel distribution per image → captures foreground/background balance."""
    flat = imgs.reshape(len(imgs), -1)
    mu = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1, keepdims=True) + 1e-8
    return ((flat - mu) ** 3).mean(axis=1) / sigma.squeeze() ** 3

def local_patch_variance(imgs, patch=4):
    """
    Average variance within non-overlapping patch×patch blocks.
    Captures local texture richness — blurry images have low local variance,
    sharp/noisy images have high local variance.
    """
    b, c, h, w = imgs.shape
    scores = []
    for img in imgs:
        img_2d = img[0]
        patches = img_2d.reshape(h // patch, patch, w // patch, patch)
        var_per_patch = patches.var(axis=(1, 3))
        scores.append(var_per_patch.mean())
    return np.array(scores)

def gradient_magnitude(imgs):
    """
    Mean absolute spatial gradient (finite differences).
    High → sharp edges; low → blurry.
    """
    dx = np.abs(np.diff(imgs, axis=-1)).reshape(len(imgs), -1).mean(axis=1)
    dy = np.abs(np.diff(imgs, axis=-2)).reshape(len(imgs), -1).mean(axis=1)
    return (dx + dy) / 2.0

# ── Compute and report ─────────────────────────────────────────────────────────
metrics = {
    "Pixel Mean":          pixel_mean,
    "Pixel Variance":      pixel_variance,
    "Pixel Skewness":      pixel_skewness,
    "Local Patch Var":     local_patch_variance,
    "Gradient Magnitude":  gradient_magnitude,
}

results = {}
print("\n" + "=" * 65)
print(f"{'Metric':<22} {'Real':>10} {'ODE':>10} {'SDE':>10}")
print("=" * 65)

for name, fn in metrics.items():
    r = fn(real_np)
    o = fn(gen_ode)
    s = fn(gen_sde)
    results[name] = {"real": r, "ode": o, "sde": s}
    print(f"{name:<22} {r.mean():>10.4f} {o.mean():>10.4f} {s.mean():>10.4f}")

print("=" * 65)
print("\nStandard deviations (spread of the distribution):")
print(f"{'Metric':<22} {'Real':>10} {'ODE':>10} {'SDE':>10}")
print("-" * 65)
for name, vals in results.items():
    print(f"{name:<22} {vals['real'].std():>10.4f} {vals['ode'].std():>10.4f} {vals['sde'].std():>10.4f}")

# ── Plot distributions ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
for ax, (name, vals) in zip(axes, results.items()):
    for label, color, data in [("Real", "black", vals["real"]),
                                ("ODE",  "blue",  vals["ode"]),
                                ("SDE",  "red",   vals["sde"])]:
        ax.hist(data, bins=40, alpha=0.5, color=color, label=label, density=True)
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("./examples/distribution_analysis.png", dpi=150)
print("\nPlot saved to ./examples/distribution_analysis.png")

# ── FID ───────────────────────────────────────────────────────────────────────
print("\nComputing Inception features for FID (10k real, 2k generated each)...")
feats_real = get_inception_features(real_np_full)   # all 10k real images
feats_ode  = get_inception_features(gen_ode_fid)    # 2k ODE samples
feats_sde  = get_inception_features(gen_sde_fid)    # 2k SDE samples

fid_ode = compute_fid(feats_real, feats_ode)
fid_sde = compute_fid(feats_real, feats_sde)

print("\n" + "=" * 40)
print(f"{'FID (lower is better)':<25}")
print(f"  ODE: {fid_ode:.2f}")
print(f"  SDE: {fid_sde:.2f}")
print("=" * 40)
