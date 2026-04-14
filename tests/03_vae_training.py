#!/usr/bin/env python
# coding: utf-8

# # Notebook 3 - VAE Training 
# 
# This notebook implements and trains the VAE architectures described in Sokol (2022), "Autoencoder Market Models for Interest Rates".
# 1. **Single-Currency VAE** : trained on one currency at a time
# 2. **Multi-Currency VAE** : trained on all currencies with shared latent space
# 3. **Multi-Currency CVAE** : conditional VAE with one-hot currency encoding
# 
# ### Architecture (Tables 1-3 in the paper):
# - **Input**: N=7 swap rates (2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y)
# - **Latent space**: K=2 dimensions
# - **Pre-processing**: linear map from [-5%, 25%] to [0, 1]
# - **Loss**: L2 reconstruction + β·KLD (β = 1e-7)
# 
# ### Extensions:
# - **Post-processing gradient descent** (Section 2.3.1)
# - **Nelson-Siegel comparison** (Section 2.4.3)
# - **In-sample vs Out-of-sample validation** (Figure 12)
# - **Paper figures replication** (Figures 9, 10, 11, 12, 13, 14, 15)
# - **Chapter 3: Q-Measure AEMM Models** (Forward Rate & Short Rate)
# - **Chapter 4: P-Measure AEMM Models** (Autoregressive)

# ## 0 - Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.optimize import minimize, least_squares
from scipy.stats import gaussian_kde
from datetime import datetime


# In[2]:


Path("figures").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# In[3]:


## 1 - Configuration & Data
SEED = 42  # reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[5]:


DATA_PATH = Path("data/df_multi.csv")
EXCLUDE_CURRENCIES = {"CHF"} 
df_long = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df_long = df_long[~df_long["currency"].isin(EXCLUDE_CURRENCIES)].copy()


# In[6]:


CURRENCIES = sorted(df_long["currency"].unique())
TARGET_TENORS = [2, 3, 5, 10, 15, 20, 30]
TENORS = np.array(TARGET_TENORS, dtype=float)
TENOR_COLS = [str(t) for t in TARGET_TENORS]    # column names in the CSV are strings


# In[7]:


# Reshape the long-format CSV into a dict[ccy -> DataFrame(dates, tenor-cols)].
swap_aligned: dict[str, pd.DataFrame] = {}
for ccy in CURRENCIES:
    sub = (df_long[df_long["currency"] == ccy]
           .set_index("Date")
           .sort_index()
           [TENOR_COLS]
           .copy())
    sub.columns = TARGET_TENORS             
    swap_aligned[ccy] = sub


# In[8]:


swap_data = swap_aligned


# In[9]:


dates_ref = swap_aligned[CURRENCIES[0]].index
print(f"Loaded {len(CURRENCIES)} currencies: {CURRENCIES}")
print(f"Dates: {len(dates_ref)}  "
      f"({dates_ref.min().date()} -> {dates_ref.max().date()})")
print(f"Tenors: {TARGET_TENORS}")
for ccy in CURRENCIES:
    print(f"  {ccy}: shape={swap_aligned[ccy].shape}")


# In[10]:


# Use all available currencies in swap_aligned
CURRENCIES = list(swap_aligned.keys())
N_CCY = len(CURRENCIES)
CCY_TO_IDX = {c: i for i, c in enumerate(CURRENCIES)}
print("Dataset used: swap_aligned")
for ccy, df in swap_aligned.items():
    print(f"  {ccy}: {df.shape[0]} obs × {df.shape[1]} tenors")


# In[ ]:





# ## 2 - Data pre-processing

# In[11]:


# Following the paper exactly:
# - Map swap rates from [S_min, S_max] = [-5%, 25%] to [0, 1] using linear transform
# - This matches the Sigmoid output activation of the decoder

S_MIN = -0.05   # lower bound (decimal)
S_MAX = 0.25    # upper bound (decimal)
BP_PER_UNIT = 10000.0


# In[12]:


def normalize_rates(rates: np.ndarray) -> np.ndarray:
    """Map swap rates (decimal) from [S_MIN, S_MAX] to [0, 1]."""
    x = (rates - S_MIN) / (S_MAX - S_MIN)
    return np.clip(x, 0.0, 1.0)


# In[13]:


def denormalize_rates(normed: np.ndarray) -> np.ndarray:
    """Map from [0, 1] back to swap rates (decimal)."""
    return normed * (S_MAX - S_MIN) + S_MIN
# Single-currency: dict of tensors
single_ccy_data: Dict[str, Dict] = {}
for ccy in CURRENCIES:
    df = swap_data[ccy]                       # full history per currency
    rates = df.values.astype(np.float32)      # decimal
    rates_norm = normalize_rates(rates).astype(np.float32)
    single_ccy_data[ccy] = {
        "raw": rates,
        "norm": rates_norm,
        "tensor": torch.from_numpy(rates_norm),
        "dates": df.index,
    }
# Multi-currency aligned dataset (shared latent space)
multi_rates = []
multi_labels = []   # one-hot for CVAE
multi_ccy_ids = []
multi_dates = []
for ccy in CURRENCIES:
    df = swap_aligned[ccy]
    rates = df.values.astype(np.float32)
    rates_norm = normalize_rates(rates).astype(np.float32)
    n = len(rates_norm)

    one_hot = np.zeros((n, N_CCY), dtype=np.float32)
    one_hot[:, CCY_TO_IDX[ccy]] = 1.0

    multi_rates.append(rates_norm)
    multi_labels.append(one_hot)
    multi_ccy_ids.extend([ccy] * n)
    multi_dates.extend(df.index.tolist())


# In[14]:


multi_rates = np.vstack(multi_rates)
multi_labels = np.vstack(multi_labels)


# In[15]:


X_multi = torch.from_numpy(multi_rates)
Y_multi = torch.from_numpy(multi_labels)


# In[16]:


print(f"\nMulti-currency dataset: {X_multi.shape[0]} samples × {X_multi.shape[1]} features")
print(f"Currency distribution: {pd.Series(multi_ccy_ids).value_counts().to_dict()}")
## 2b - Train/Test Split for Out-of-Sample Validation


# In[17]:


# Following the paper (Section 2.4.2). Adjust `TRAIN_CUTOFF` to match your data range.
# TRAIN_CUTOFF = pd.Timestamp("2025-07-31") # 80 train / 20 tests
# TRAIN_CUTOFF = pd.Timestamp("2024-07-31") # 50 train / 50 tests
TRAIN_CUTOFF = pd.Timestamp("2025-03-31") # split 70/30 (recommandée pour ton cas)


# In[18]:


def create_train_test_split(swap_aligned: Dict, cutoff_date: pd.Timestamp):
    """Split aligned swap data into train (<=cutoff) and test (>cutoff)."""
    train_data, test_data = {}, {}
    for ccy, df in swap_aligned.items():
        train_data[ccy] = df[df.index <= cutoff_date]
        test_data[ccy]  = df[df.index >  cutoff_date]
    return train_data, test_data


# In[19]:


swap_aligned_train, swap_aligned_test = create_train_test_split(swap_aligned, TRAIN_CUTOFF)


# In[20]:


print(f"Train/Test split at {TRAIN_CUTOFF.strftime('%Y-%m-%d')}")
print("\nTraining set:")
for ccy, df in swap_aligned_train.items():
    if len(df) > 0:
        print(f"  {ccy}: {len(df)} obs ({df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')})")
    else:
        print(f"  {ccy}: 0 observations")


# In[21]:


print("\nTest set:")
for ccy, df in swap_aligned_test.items():
    if len(df) > 0:
        print(f"  {ccy}: {len(df)} obs ({df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')})")
    else:
        print(f"  {ccy}: 0 observations")
def prepare_multi_currency_tensors(swap_dict: Dict, currencies: List[str]):
    """Prepare normalized tensors for multi-currency training/testing."""
    rates_list, labels_list, ccy_ids, dates_list = [], [], [], []
    n_ccy = len(currencies)
    ccy_to_idx = {c: i for i, c in enumerate(currencies)}

    for ccy in currencies:
        df = swap_dict[ccy]
        if len(df) == 0:
            continue
        rates = df.values.astype(np.float32)
        rates_norm = normalize_rates(rates).astype(np.float32)
        n = len(rates_norm)

        one_hot = np.zeros((n, n_ccy), dtype=np.float32)
        one_hot[:, ccy_to_idx[ccy]] = 1.0

        rates_list.append(rates_norm)
        labels_list.append(one_hot)
        ccy_ids.extend([ccy] * n)
        dates_list.extend(df.index.tolist())

    X = torch.from_numpy(np.vstack(rates_list))
    Y = torch.from_numpy(np.vstack(labels_list))
    return X, Y, ccy_ids, dates_list


# In[22]:


X_train, Y_train, train_ccy_ids, train_dates = prepare_multi_currency_tensors(swap_aligned_train, CURRENCIES)
if sum(len(df) for df in swap_aligned_test.values()) > 0:
    X_test, Y_test, test_ccy_ids, test_dates = prepare_multi_currency_tensors(swap_aligned_test, CURRENCIES)
else:
    X_test = torch.empty(0, 7)
    Y_test = torch.empty(0, N_CCY)
    test_ccy_ids, test_dates = [], []


# In[23]:


print(f"\nTraining tensors: X_train {tuple(X_train.shape)}, Y_train {tuple(Y_train.shape)}")
print(f"Test tensors:     X_test  {tuple(X_test.shape)},  Y_test  {tuple(Y_test.shape)}")


# ## 3 - VAE Architecture

# In[24]:


# We implement all three architectures from Tables 1-3 of the paper.
class VAE(nn.Module):
    """
    VAE for swap curves.
    - single-currency: Table 1 style
    - multi-currency: Table 2 style (deeper decoder)
    """

    def __init__(self, input_dim: int = 7, latent_dim: int = 2, hidden_dim: int = 4, multi_currency: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.multi_currency = multi_currency

        if not multi_currency:
            # Single-currency (Table 1 style)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2 * latent_dim),  # [mu | logvar]
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid(),
            )
        else:
            # Multi-currency (Table 2 style)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, 2 * latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid(),
            )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_latent(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x.to(next(self.parameters()).device))
        return mu.cpu().numpy()
class CVAE(nn.Module):
    """
    Conditional VAE (Table 3 style): one-hot currency concatenated to encoder/decoder inputs.
    """

    def __init__(self, input_dim: int = 7, latent_dim: int = 2, n_currencies: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_currencies = n_currencies

        enc_in = input_dim + n_currencies
        dec_in = latent_dim + n_currencies

        self.encoder = nn.Sequential(
            nn.Linear(enc_in, enc_in - 1),
            nn.Tanh(),
            nn.Linear(enc_in - 1, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, y], dim=-1))
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, y], dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

    def get_latent(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(
                x.to(next(self.parameters()).device),
                y.to(next(self.parameters()).device),
            )
        return mu.cpu().numpy()
print("\n Model classes defined")
print(f"   VAE single-ccy params: {sum(p.numel() for p in VAE(7, 2).parameters()):,}")
print(f"   VAE multi-ccy params:  {sum(p.numel() for p in VAE(7, 2, multi_currency=True).parameters()):,}")
print(f"   CVAE params:           {sum(p.numel() for p in CVAE(7, 2, N_CCY).parameters()):,}")


# ## 4 - Loss function (FIXED)

# From Eq. (5) of the paper:
# $$\mathcal{L}_{VAE} = \frac{1}{|\text{batch}|}\sum_{i \in \text{batch}} \sum_{n=1}^{N}(S_{i,n}-S'_{i,n})^2 + \beta \cdot D_{KLD}(\mu, \sigma)$$

# In[25]:


# reconstruction error must be SUMMED over tenors, then MEAN over the batch.
def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss as defined in Eq. (5) of Sokol (2022).

    Reconstruction: sum of squared errors over tenors, then mean over batch.
    KLD: standard Gaussian KL, summed over latent dims then mean over batch.
    """
    # L2 reconstruction: sum over tenors, mean over batch
    recon_per_sample = torch.sum((x_recon - x) ** 2, dim=-1)  # (batch,)
    recon_loss = torch.mean(recon_per_sample)

    # KLD
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
    kld_loss = torch.mean(kld_per_sample)

    total = recon_loss + beta * kld_loss
    return total, recon_loss, kld_loss


# ## 5 - Training loop

# In[26]:


# Includes gradient clipping, `min_lr` floor, and a plateau scheduler with generous patience.
def train_vae(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: Optional[torch.Tensor] = None,
    n_epochs: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 1e-7,
    print_every: int = 500,
    scheduler_patience: int = 500,
    grad_clip: float = 1.0,
) -> Dict[str, list]:
    model.to(device)
    is_cvae = isinstance(model, CVAE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=0.5, min_lr=1e-6
    )

    if is_cvae:
        assert Y_train is not None, "CVAE requires Y_train one-hot labels"
        dataset = TensorDataset(X_train, Y_train)
    else:
        dataset = TensorDataset(X_train)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = {"total": [], "recon": [], "kld": [], "lr": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        tot = rec = kldv = 0.0
        n_batches = 0

        for batch in loader:
            if is_cvae:
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                x_recon, mu, logvar = model(x_batch, y_batch)
            else:
                x_batch = batch[0].to(device)
                x_recon, mu, logvar = model(x_batch)

            loss, recon, kld = vae_loss(x_recon, x_batch, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            tot  += float(loss.item())
            rec  += float(recon.item())
            kldv += float(kld.item())
            n_batches += 1

        tot  /= n_batches
        rec  /= n_batches
        kldv /= n_batches

        history["total"].append(tot)
        history["recon"].append(rec)
        history["kld"].append(kldv)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(tot)

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"  Epoch {epoch:>5d}/{n_epochs} | "
                f"Loss: {tot:.6e} | Recon: {rec:.6e} | KLD: {kldv:.3e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e}"
            )

    model.eval()
    return history


# ## 6 - Train all three architectures

# In[27]:


print("\n" + "=" * 72)
print("Training SINGLE-CURRENCY VAEs")
print("=" * 72)
single_vae_models: Dict[str, VAE] = {}
single_vae_histories: Dict[str, Dict] = {}


# In[28]:


for ccy in CURRENCIES:
    print(f"\n{'─'*50}\n  {ccy}\n{'─'*50}")
    model = VAE(input_dim=7, latent_dim=2, hidden_dim=4, multi_currency=False)
    X = single_ccy_data[ccy]["tensor"]
    hist = train_vae(
        model, X,
        n_epochs=5000, batch_size=128, lr=1e-3, beta=1e-7,
        print_every=500, scheduler_patience=500,
    )
    single_vae_models[ccy] = model
    single_vae_histories[ccy] = hist
print("\n" + "=" * 72)
print("Training MULTI-CURRENCY VAE (shared latent space)")
print("=" * 72)
multi_vae = VAE(input_dim=7, latent_dim=2, hidden_dim=4, multi_currency=True)
multi_vae_history = train_vae(
    multi_vae, X_multi,
    n_epochs=5000, batch_size=256, lr=1e-3, beta=1e-7,
    print_every=500, scheduler_patience=500,
)
print("\n" + "=" * 72)
print("Training MULTI-CURRENCY CVAE (conditional)")
print("=" * 72)
cvae = CVAE(input_dim=7, latent_dim=2, n_currencies=N_CCY)
cvae_history = train_vae(
    cvae, X_multi, Y_train=Y_multi,
    n_epochs=5000, batch_size=256, lr=1e-3, beta=1e-7,
    print_every=500, scheduler_patience=500,
)


# ### 6b - Post-Processing Gradient Descent (Section 2.3.1)

# From the paper:
# > "A post-processing step increases the accuracy of VAE mapping by performing gradient
# > descent minimizing L2 loss starting from the center μ(S) of the distribution produced
# > by the encoder."

# In[29]:


#This refines the latent representation for each observation by optimizing z directly to minimize reconstruction error.
def post_processing_gradient_descent(
    model: nn.Module,
    X_norm: torch.Tensor,
    Y_labels: Optional[torch.Tensor] = None,
    n_steps: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starting from μ(S) produced by the encoder, perform gradient descent on z
    to minimize L2 reconstruction loss.

    Returns (z_refined, x_recon_refined) as numpy arrays.
    """
    model.eval()
    is_cvae = isinstance(model, CVAE)

    X_dev = X_norm.to(device)
    Y_dev = Y_labels.to(device) if is_cvae else None

    with torch.no_grad():
        if is_cvae:
            mu_init, _ = model.encode(X_dev, Y_dev)
        else:
            mu_init, _ = model.encode(X_dev)

    z = mu_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        x_recon = model.decode(z, Y_dev) if is_cvae else model.decode(z)
        # Consistent with Eq. 5: sum over tenors, mean over batch
        loss = torch.mean(torch.sum((x_recon - X_dev) ** 2, dim=-1))
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{n_steps}: L2 loss = {loss.item():.6e}")

    with torch.no_grad():
        x_recon_final = model.decode(z, Y_dev) if is_cvae else model.decode(z)

    return z.detach().cpu().numpy(), x_recon_final.cpu().numpy()
print("\n" + "=" * 72)
print("Applying Post-Processing Gradient Descent")
print("=" * 72)


# In[30]:


print("\nMulti-Currency VAE:")
z_refined_multi, x_recon_refined_multi = post_processing_gradient_descent(
    multi_vae, X_multi, n_steps=100, lr=0.01, verbose=True
)


# In[31]:


print("\nCVAE:")
z_refined_cvae, x_recon_refined_cvae = post_processing_gradient_descent(
    cvae, X_multi, Y_multi, n_steps=100, lr=0.01, verbose=True
)
# Compare RMSE before vs after post-processing
def compute_rmse_from_recon(x_recon_norm: np.ndarray, x_true_norm: np.ndarray) -> np.ndarray:
    """RMSE in basis points from normalized values."""
    x_recon_dec = denormalize_rates(x_recon_norm)
    x_true_dec  = denormalize_rates(x_true_norm)
    return np.sqrt(np.mean((x_recon_dec - x_true_dec) ** 2, axis=1)) * BP_PER_UNIT


# In[32]:


multi_vae.eval()
with torch.no_grad():
    x_recon_before, _, _ = multi_vae(X_multi.to(device))
    x_recon_before = x_recon_before.cpu().numpy()


# In[33]:


rmse_before = compute_rmse_from_recon(x_recon_before, X_multi.numpy())
rmse_after  = compute_rmse_from_recon(x_recon_refined_multi, X_multi.numpy())


# In[34]:


print("\nMulti-Currency VAE RMSE Comparison:")
print(f"  Before post-processing: Mean={np.mean(rmse_before):.2f} bp, Median={np.median(rmse_before):.2f} bp")
print(f"  After  post-processing: Mean={np.mean(rmse_after):.2f} bp, Median={np.median(rmse_after):.2f} bp")
print(f"  Improvement: {(1 - np.mean(rmse_after)/np.mean(rmse_before))*100:.1f}%")


# ## 7 - Nelson-Siegel Implementation (Section 2.2)

# Classical Nelson-Siegel basis with 3 factors:
# $$S(\tau) = \beta_1 + \beta_2 \frac{1-e^{-\tau/\lambda}}{\tau/\lambda} + \beta_3 \left( \frac{1-e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda} \right)$$
# 

# In[35]:


class NelsonSiegel:
    """Nelson-Siegel curve fitting with fixed lambda."""

    def __init__(self, lambda_param: float = 1.5):
        self.lambda_param = lambda_param

    def basis_functions(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        lam = self.lambda_param
        tau_lam = np.maximum(tau / lam, 1e-10)

        f1 = np.ones_like(tau)
        f2 = (1 - np.exp(-tau_lam)) / tau_lam
        f3 = f2 - np.exp(-tau_lam)
        return np.column_stack([f1, f2, f3])

    def fit(self, tau: np.ndarray, rates: np.ndarray) -> np.ndarray:
        X = self.basis_functions(tau)
        beta, _, _, _ = np.linalg.lstsq(X, rates, rcond=None)
        return beta

    def predict(self, tau: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return self.basis_functions(tau) @ beta

    def fit_multiple(self, tau: np.ndarray, rates_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_obs = rates_matrix.shape[0]
        betas  = np.zeros((n_obs, 3))
        fitted = np.zeros_like(rates_matrix)
        for i in range(n_obs):
            betas[i]  = self.fit(tau, rates_matrix[i])
            fitted[i] = self.predict(tau, betas[i])
        return betas, fitted


# In[36]:


print("\n" + "=" * 72)
print("Fitting Nelson-Siegel Model")
print("=" * 72)
ns_model = NelsonSiegel(lambda_param=1.5)
tenors = np.array(TARGET_TENORS, dtype=float)
ns_results: Dict[str, Dict] = {}
for ccy in CURRENCIES:
    rates = swap_aligned[ccy].values
    betas, fitted = ns_model.fit_multiple(tenors, rates)
    rmse_bp = np.sqrt(np.mean((fitted - rates) ** 2, axis=1)) * BP_PER_UNIT
    ns_results[ccy] = {"betas": betas, "fitted": fitted, "rmse_bp": rmse_bp}
    print(f"  {ccy}: Mean RMSE = {np.mean(rmse_bp):.2f} bp, Median = {np.median(rmse_bp):.2f} bp")


# ## 8 - Plot training convergence

# In[37]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
for ccy in CURRENCIES:
    ax.plot(single_vae_histories[ccy]["total"], label=ccy, alpha=0.8)
ax.set_title("Single-Currency VAE", fontsize=13)
ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss")
ax.set_yscale("log"); ax.legend()

ax = axes[1]
ax.plot(multi_vae_history["total"], label="Total", alpha=0.9)
ax.plot(multi_vae_history["recon"], label="Recon", alpha=0.7)
ax.set_title("Multi-Currency VAE", fontsize=13)
ax.set_xlabel("Epoch"); ax.set_yscale("log"); ax.legend()

ax = axes[2]
ax.plot(cvae_history["total"], label="Total", alpha=0.9)
ax.plot(cvae_history["recon"], label="Recon", alpha=0.7)
ax.set_title("Multi-Currency CVAE", fontsize=13)
ax.set_xlabel("Epoch"); ax.set_yscale("log"); ax.legend()

plt.suptitle("Training Convergence", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig_08_training_loss.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 9 - Compute reconstruction RMSE for all models

# Single-VAE is now evaluated only on the aligned dates so it is directly comparable
# to Multi-VAE / CVAE / Nelson-Siegel (which all live on `swap_aligned`).

# In[38]:


def compute_rmse_bp(model: nn.Module, X_norm: torch.Tensor, Y_labels: Optional[torch.Tensor] = None):
    """Per-observation reconstruction RMSE in basis points."""
    model.eval()
    with torch.no_grad():
        X_dev = X_norm.to(device)
        if isinstance(model, CVAE):
            assert Y_labels is not None
            Y_dev = Y_labels.to(device)
            X_recon, _, _ = model(X_dev, Y_dev)
        else:
            X_recon, _, _ = model(X_dev)

    X_recon_np = X_recon.cpu().numpy()
    X_true_np  = X_norm.cpu().numpy()

    x_recon = denormalize_rates(X_recon_np)
    x_true  = denormalize_rates(X_true_np)

    return np.sqrt(np.mean((x_recon - x_true) ** 2, axis=1)) * BP_PER_UNIT, x_recon
rmse_results: Dict[str, np.ndarray] = {}


# In[39]:


# Single-currency VAE — evaluate only on aligned dates for a fair comparison
for ccy in CURRENCIES:
    aligned_dates = swap_aligned[ccy].index
    full_dates    = single_ccy_data[ccy]["dates"]
    mask = full_dates.isin(aligned_dates)
    X_aligned = single_ccy_data[ccy]["tensor"][mask]
    rmse_bp, _ = compute_rmse_bp(single_vae_models[ccy], X_aligned)
    rmse_results[f"SingleVAE_{ccy}"] = rmse_bp


# In[40]:


# Multi-currency VAE (per-currency slices)
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    X_slice = X_multi[offset:offset+n]
    rmse_bp, _ = compute_rmse_bp(multi_vae, X_slice)
    rmse_results[f"MultiVAE_{ccy}"] = rmse_bp
    offset += n


# In[41]:


# CVAE (per-currency slices)
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    X_slice = X_multi[offset:offset+n]
    Y_slice = Y_multi[offset:offset+n]
    rmse_bp, _ = compute_rmse_bp(cvae, X_slice, Y_slice)
    rmse_results[f"CVAE_{ccy}"] = rmse_bp
    offset += n


# In[42]:


# Nelson-Siegel
for ccy in CURRENCIES:
    rmse_results[f"NelsonSiegel_{ccy}"] = ns_results[ccy]["rmse_bp"]


# In[43]:


print(f"\n{'Model':<20} {'Currency':<8} {'Mean(bp)':>10} {'Median':>10} {'95th':>10}")
print("─" * 64)
for key, rmse in rmse_results.items():
    model_name, ccy = key.rsplit("_", 1)
    print(f"{model_name:<20} {ccy:<8} {np.mean(rmse):>10.2f} {np.median(rmse):>10.2f} {np.percentile(rmse, 95):>10.2f}")


# ## 10 - Figure 9: RMSE Distribution by Model Type

# In[44]:


rmse_single_all = np.concatenate([rmse_results[f"SingleVAE_{ccy}"] for ccy in CURRENCIES])
rmse_multi_all  = np.concatenate([rmse_results[f"MultiVAE_{ccy}"]  for ccy in CURRENCIES])
rmse_cvae_all   = np.concatenate([rmse_results[f"CVAE_{ccy}"]      for ccy in CURRENCIES])


# In[45]:


fig, ax = plt.subplots(figsize=(10, 6))
x_range = np.linspace(0, 50, 500)

for data, label, color in [
    (rmse_multi_all,  "2D Multi-Currency VAE",  "tab:blue"),
    (rmse_cvae_all,   "2D Multi-Currency CVAE", "tab:orange"),
    (rmse_single_all, "2D Single-Currency VAE", "tab:green"),
]:
    kde = gaussian_kde(data, bw_method=0.3)
    ax.fill_between(x_range, kde(x_range), alpha=0.5, label=label, color=color)
    ax.plot(x_range, kde(x_range), color=color, lw=1.5)
    
ax.set_xlabel("Swap Rate RMSE (bp)"); ax.set_ylabel("Probability Density")
ax.set_xlim(0, 50); ax.set_ylim(0, 0.2)
ax.legend(loc="upper right"); ax.set_title("Figure 9: Distribution of In-Sample RMSE by VAE Architecture")
plt.tight_layout()
plt.savefig("figures/fig_09_rmse_by_model.png", dpi=150, bbox_inches="tight")
plt.show()


# In[46]:


print("\nSummary Statistics:")
print(f"  Single-Currency VAE: Mean={np.mean(rmse_single_all):.1f} bp, Median={np.median(rmse_single_all):.1f} bp")
print(f"  Multi-Currency VAE:  Mean={np.mean(rmse_multi_all):.1f} bp,  Median={np.median(rmse_multi_all):.1f} bp")
print(f"  Multi-Currency CVAE: Mean={np.mean(rmse_cvae_all):.1f} bp,  Median={np.median(rmse_cvae_all):.1f} bp")



# ## 11 - Figure 10: VAE vs Nelson-Siegel Comparison

# In[47]:


rmse_ns_all = np.concatenate([rmse_results[f"NelsonSiegel_{ccy}"] for ccy in CURRENCIES])

fig, ax = plt.subplots(figsize=(10, 6))
x_range = np.linspace(0, 50, 500)

for data, label, color in [
    (rmse_ns_all,    "3D Nelson-Siegel",       "tab:purple"),
    (rmse_cvae_all,  "2D Multi-Currency CVAE", "tab:orange"),
    (rmse_multi_all, "2D Multi-Currency VAE",  "tab:blue"),
]:
    kde = gaussian_kde(data, bw_method=0.3)
    ax.fill_between(x_range, kde(x_range), alpha=0.5, label=label, color=color)
    ax.plot(x_range, kde(x_range), color=color, lw=1.5)
ax.set_xlabel("Swap Rate RMSE (bp)"); ax.set_ylabel("Probability Density")
ax.set_xlim(0, 50); ax.set_ylim(0, 0.2)
ax.legend(loc="upper right"); ax.set_title("Figure 10: 2D VAE vs 3D Nelson-Siegel Comparison")
plt.tight_layout()
plt.savefig("figures/fig_10_vae_vs_ns.png", dpi=150, bbox_inches="tight")
plt.show()


# In[48]:


print("\nKey Finding:")
print(f"  Nelson-Siegel (3D):      Mean={np.mean(rmse_ns_all):.1f} bp")
print(f"  Multi-Currency VAE (2D): Mean={np.mean(rmse_multi_all):.1f} bp")


# ## 12 - Figure 11: RMSE Distribution by Currency
# 

# In[49]:


fig, ax = plt.subplots(figsize=(12, 6))
x_range = np.linspace(0, 50, 500)
cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(CURRENCIES)))

for i, ccy in enumerate(CURRENCIES):
    rmse = rmse_results[f"MultiVAE_{ccy}"]
    kde = gaussian_kde(rmse, bw_method=0.3)
    ax.fill_between(x_range, kde(x_range), alpha=0.4, label=ccy, color=cmap_colors[i])
    ax.plot(x_range, kde(x_range), color=cmap_colors[i], lw=1.5)
    
ax.set_xlabel("Swap Rate RMSE (bp)"); ax.set_ylabel("Probability Density")
ax.set_xlim(0, 50); ax.set_ylim(0, 0.3)
ax.legend(loc="upper right", ncol=2)
ax.set_title("Figure 11: 2D Multi-Currency VAE RMSE by Currency")
plt.tight_layout()
plt.savefig("figures/fig_11_rmse_by_currency.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 13 - Figure 12: In-Sample vs Out-of-Sample Validation
# 

# In[50]:


print("\n" + "=" * 72)
print("Training Multi-Currency VAE for Out-of-Sample Validation")
print("=" * 72)
multi_vae_oos = VAE(input_dim=7, latent_dim=2, hidden_dim=4, multi_currency=True)
_ = train_vae(
    multi_vae_oos, X_train,
    n_epochs=5000, batch_size=256, lr=1e-3, beta=1e-7,
    print_every=500, scheduler_patience=500,
)
rmse_insample, _ = compute_rmse_bp(multi_vae_oos, X_train)
if len(X_test) > 0:
    rmse_oos, _ = compute_rmse_bp(multi_vae_oos, X_test)
else:
    rmse_oos = np.array([])
print(f"\nIn-sample:     {len(rmse_insample)} observations, Mean RMSE = {np.mean(rmse_insample):.2f} bp")
if len(rmse_oos) > 0:
    print(f"Out-of-sample: {len(rmse_oos)} observations, Mean RMSE = {np.mean(rmse_oos):.2f} bp")
else:
    print("Out-of-sample: no observations (adjust TRAIN_CUTOFF).")
    
fig, ax = plt.subplots(figsize=(10, 6))
x_range = np.linspace(0, 50, 500)
kde_in = gaussian_kde(rmse_insample, bw_method=0.3)
ax.fill_between(x_range, kde_in(x_range), alpha=0.5, label="In-sample", color="tab:blue")
ax.plot(x_range, kde_in(x_range), color="tab:blue", lw=1.5)

if len(rmse_oos) > 0:
    kde_oos = gaussian_kde(rmse_oos, bw_method=0.3)
    ax.fill_between(x_range, kde_oos(x_range), alpha=0.5, label="Out-of-sample", color="tab:orange")
    ax.plot(x_range, kde_oos(x_range), color="tab:orange", lw=1.5)
    
ax.set_xlabel("Swap Rate RMSE (bp)"); ax.set_ylabel("Probability Density")
ax.set_xlim(0, 50); ax.set_ylim(0, 0.15)
ax.legend(loc="upper right")
ax.set_title("Figure 12: 2D Multi-Currency VAE — In-Sample vs Out-of-Sample")
plt.tight_layout()
plt.savefig("figures/fig_12_insample_vs_oos.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 14 - Figure 13: World Map of Latent Space

# In[51]:


# Data-driven axis limits (no hard-coded `xlim`/`ylim`) so the plot adapts to whatever
# the encoder actually produces.
default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = {ccy: default_cycle[i % len(default_cycle)] for i, ccy in enumerate(CURRENCIES)}


# In[52]:


# Gather latent codes per currency
latent_by_ccy = {}
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    latent_by_ccy[ccy] = multi_vae.get_latent(X_multi[offset:offset+n])
    offset += n


# In[53]:


all_z = np.vstack(list(latent_by_ccy.values()))


# In[54]:


def axis_lim(vals, margin=0.15):
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


# In[55]:


z1_lim = axis_lim(all_z[:, 0])
z2_lim = axis_lim(all_z[:, 1])


# In[56]:


fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# (a) Historical scatter
ax = axes[0]
for ccy in CURRENCIES:
    z = latent_by_ccy[ccy]
    ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.3, label=ccy, color=colors[ccy])
ax.set_xlabel("$z_1$", fontsize=13); ax.set_ylabel("$z_2$", fontsize=13)
ax.set_title("(a) Historical observations in latent space", fontsize=13)
ax.legend(markerscale=4, fontsize=11)
ax.set_xlim(*z1_lim); ax.set_ylim(*z2_lim)

# (b) 2σ ellipses per currency
ax = axes[1]
for ccy in CURRENCIES:
    z = latent_by_ccy[ccy]
    mean = z.mean(axis=0)
    cov  = np.cov(z.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    angle  = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    width  = 2 * 2 * np.sqrt(evals[1])
    height = 2 * 2 * np.sqrt(evals[0])

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  fill=False, lw=2.5, color=colors[ccy], label=ccy)
    ax.add_patch(ell)
    ax.plot(mean[0], mean[1], "x", ms=10, mew=2.5, color=colors[ccy])
    ax.annotate(ccy, xy=mean, xytext=(0, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", color=colors[ccy], ha="center")
    
ax.set_xlabel("$z_1$", fontsize=13); ax.set_ylabel("$z_2$", fontsize=13)
ax.set_title("(b) 2σ covariance ellipses per currency", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(*z1_lim); ax.set_ylim(*z2_lim)
ax.autoscale_view()

fig.suptitle("Figure 13: World Map of Latent Space — Multi-Currency VAE", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig_13_world_map.png", dpi=150, bbox_inches="tight")
plt.show()


# In[57]:


print("\nLatent code summary (mean ± std):")
print(f"  {'':6} {'z1 mean':>9} {'z1 std':>9} | {'z2 mean':>9} {'z2 std':>9}")
for ccy, z in latent_by_ccy.items():
    print(f"  {ccy:6} {z[:,0].mean():>9.4f} {z[:,0].std():>9.4f} | {z[:,1].mean():>9.4f} {z[:,1].std():>9.4f}")


# ## 15 - Figure 14: Historical vs Reconstructed Curves (Spaghetti Plots)
# 

# In[58]:


tenors = np.array(TARGET_TENORS, dtype=float)
representative_ccys = CURRENCIES[:3] if len(CURRENCIES) >= 3 else CURRENCIES
fig, axes = plt.subplots(len(representative_ccys), 2, figsize=(16, 5 * len(representative_ccys)), squeeze=False)

offset = 0
for ccy in CURRENCIES:
    if ccy not in representative_ccys:
        offset += len(swap_aligned[ccy])
        continue

    row = representative_ccys.index(ccy)
    n = len(swap_aligned[ccy])
    X_slice = X_multi[offset:offset+n]

    historical = denormalize_rates(X_slice.numpy()) * 100

    multi_vae.eval()
    with torch.no_grad():
        X_recon, _, _ = multi_vae(X_slice.to(device))
    reconstructed = denormalize_rates(X_recon.cpu().numpy()) * 100

    ax = axes[row, 0]
    for i in range(0, n, max(1, n // 100)):
        ax.plot(tenors, historical[i], alpha=0.3, lw=0.5, color=colors[ccy])
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Swap Rate (%)")
    ax.set_title(f"{ccy} — Historical")
    ax.set_xlim(0, 35)

    ax = axes[row, 1]
    for i in range(0, n, max(1, n // 100)):
        ax.plot(tenors, reconstructed[i], alpha=0.3, lw=0.5, color=colors[ccy])
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Swap Rate (%)")
    ax.set_title(f"{ccy} — Reconstructed")
    ax.set_xlim(0, 35)

    ymin = min(historical.min(), reconstructed.min())
    ymax = max(historical.max(), reconstructed.max())
    axes[row, 0].set_ylim(ymin - 0.5, ymax + 0.5)
    axes[row, 1].set_ylim(ymin - 0.5, ymax + 0.5)

    offset += n
    
plt.suptitle("Figure 14: Historical vs Reconstructed Swap Rates", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_14_spaghetti.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 16 - Figure 15: Generated Curves from Latent Space Perimeter

# In[59]:


# Data-driven limits and a z₂-coloured palette, as in the simple notebook.
all_z_bg = np.vstack(list(latent_by_ccy.values()))
z1_range = all_z_bg[:, 0].max() - all_z_bg[:, 0].min()
z2_range = all_z_bg[:, 1].max() - all_z_bg[:, 1].min()
z1_lim_15 = (all_z_bg[:, 0].min() - 0.15 * z1_range, all_z_bg[:, 0].max() + 0.15 * z1_range)
z2_lim_15 = (all_z_bg[:, 1].min() - 0.10 * z2_range, all_z_bg[:, 1].max() + 0.10 * z2_range)

fig, axes = plt.subplots(len(CURRENCIES), 2, figsize=(16, 5 * len(CURRENCIES)), squeeze=False)

for row, ccy in enumerate(CURRENCIES):
    z_all = latent_by_ccy[ccy]
    mean  = z_all.mean(axis=0)
    cov   = np.cov(z_all.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)

    thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    ellipse_pts = np.array([
        mean + 2 * evecs @ (np.sqrt(evals) * np.array([np.cos(th), np.sin(th)]))
        for th in thetas
    ])

    # Left: latent space
    axL = axes[row, 0]
    axL.scatter(all_z_bg[:, 0], all_z_bg[:, 1], s=1, alpha=0.07, color="gray")
    axL.scatter(z_all[:, 0], z_all[:, 1], s=4, alpha=0.3, color=colors[ccy], label=ccy)
    axL.scatter(ellipse_pts[:, 0], ellipse_pts[:, 1],
                s=30, alpha=0.9, color=colors[ccy], zorder=5, label="2σ perimeter")
    axL.plot(mean[0], mean[1], "x", ms=12, mew=2.5, color="black", zorder=6)
    axL.set_title(f"{ccy} — Latent Space", fontsize=12, fontweight="bold")
    axL.set_xlabel("$z_1$"); axL.set_ylabel("$z_2$")
    axL.set_xlim(*z1_lim_15); axL.set_ylim(*z2_lim_15)
    axL.legend(fontsize=9, markerscale=2)

    # Right: decoded curves, coloured by z₂
    axR = axes[row, 1]
    z_tensor = torch.from_numpy(ellipse_pts.astype(np.float32)).to(device)
    with torch.no_grad():
        decoded_norm = multi_vae.decode(z_tensor).cpu().numpy()
    decoded_pct = denormalize_rates(decoded_norm) * 100

    cmap = plt.get_cmap("RdYlBu")
    z2_vals = ellipse_pts[:, 1]
    z2_norm = (z2_vals - z2_vals.min()) / ((z2_vals.max() - z2_vals.min()) + 1e-12)
    for i in range(decoded_pct.shape[0]):
        axR.plot(tenors, decoded_pct[i], lw=1.4, alpha=0.8, color=cmap(z2_norm[i]))

    # Overlay a sample of actual market curves as faint gray reference
    raw_pct = swap_aligned[ccy].values * 100
    for idx in np.linspace(0, len(raw_pct) - 1, 10, dtype=int):
        axR.plot(tenors, raw_pct[idx], lw=0.5, alpha=0.25, color="gray", zorder=1)

    axR.set_title(f"{ccy} — Decoded Curves (2σ ellipse perimeter)", fontsize=12, fontweight="bold")
    axR.set_xlabel("Maturity (years)"); axR.set_ylabel("Swap Rate (%)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(z2_vals.min(), z2_vals.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=axR, label="$z_2$ value", shrink=0.7)

plt.suptitle("Figure 15: Curves Generated by Decoding Points on 2σ Ellipse Perimeter", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_15_ellipse_curves.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 17 - Forward Rate AEMM (Section 3.1.3)

# Chapter 3: Q-Measure AEMM Models
# 1. **Forward Rate AEMM** (Section 3.1.3)
# 2. **Multi-Factor Short Rate AEMM** (Section 3.2.2)
# 

# In[60]:


class ForwardRateAEMM:
    """
    Forward Rate Autoencoder Market Model.

    Models the evolution of swap curves in VAE latent space with optional
    re-encoding to keep curves on the learned manifold.
    """

    def __init__(self, vae_model: VAE, tenors: np.ndarray):
        self.vae = vae_model
        self.tenors = tenors
        self.device = next(vae_model.parameters()).device

    def encode(self, swap_rates: np.ndarray) -> np.ndarray:
        rates_norm = normalize_rates(swap_rates).astype(np.float32)
        X = torch.from_numpy(rates_norm).to(self.device)
        return self.vae.get_latent(X)

    def decode(self, z: np.ndarray) -> np.ndarray:
        self.vae.eval()
        z_tensor = torch.from_numpy(z.astype(np.float32)).to(self.device)
        with torch.no_grad():
            decoded_norm = self.vae.decode(z_tensor).cpu().numpy()
        return denormalize_rates(decoded_norm)

    def compute_jacobian(self, z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Finite-difference Jacobian of the decoder at point z."""
        latent_dim = z.shape[-1]
        n_tenors = len(self.tenors)
        jacobian = np.zeros((n_tenors, latent_dim))
        for k in range(latent_dim):
            z_plus, z_minus = z.copy(), z.copy()
            z_plus[k]  += eps
            z_minus[k] -= eps
            s_plus  = self.decode(z_plus.reshape(1, -1))[0]
            s_minus = self.decode(z_minus.reshape(1, -1))[0]
            jacobian[:, k] = (s_plus - s_minus) / (2 * eps)
        return jacobian

    def simulate_paths(
        self,
        initial_curve: np.ndarray,
        n_paths: int,
        n_steps: int,
        dt: float,
        volatility: np.ndarray,
        correlation: Optional[np.ndarray] = None,
        re_encode: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        latent_dim = self.vae.latent_dim
        n_tenors = len(self.tenors)

        if correlation is None:
            correlation = np.eye(latent_dim)
        L = np.linalg.cholesky(correlation)

        z_init = self.encode(initial_curve.reshape(1, -1))[0]

        z_paths = np.zeros((n_paths, n_steps + 1, latent_dim))
        curve_paths = np.zeros((n_paths, n_steps + 1, n_tenors))
        z_paths[:, 0, :] = z_init
        curve_paths[:, 0, :] = initial_curve

        sqrt_dt = np.sqrt(dt)

        for t in range(n_steps):
            dW = np.random.randn(n_paths, latent_dim) @ L.T
            for p in range(n_paths):
                z_current = z_paths[p, t, :]
                diffusion = volatility * dW[p] * sqrt_dt
                z_new = z_current + diffusion
                curve_new = self.decode(z_new.reshape(1, -1))[0]
                if re_encode:
                    z_new = self.encode(curve_new.reshape(1, -1))[0]
                z_paths[p, t + 1, :] = z_new
                curve_paths[p, t + 1, :] = curve_new

        return z_paths, curve_paths


# In[61]:


print("\n" + "=" * 72)
print("Testing Forward Rate AEMM")
print("=" * 72)

fr_aemm = ForwardRateAEMM(multi_vae, tenors)
initial_curve = swap_aligned[CURRENCIES[0]].iloc[0].values
print(f"\nInitial curve ({CURRENCIES[0]}): {initial_curve * 100}")

volatility = np.array([0.01, 0.01])
z_paths, curve_paths = fr_aemm.simulate_paths(
    initial_curve=initial_curve,
    n_paths=100,
    n_steps=12,
    dt=1/12,
    volatility=volatility,
    re_encode=True,
)
print(f"\nSimulated {z_paths.shape[0]} paths over {z_paths.shape[1]-1} time steps")
print(f"Latent paths shape: {z_paths.shape}")
print(f"Curve paths shape:  {curve_paths.shape}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for p in range(min(20, z_paths.shape[0])):
    ax.plot(z_paths[p, :, 0], z_paths[p, :, 1], alpha=0.5, lw=0.8)
    ax.scatter(z_paths[p, 0, 0],  z_paths[p, 0, 1],  s=30, c='green', marker='o')
    ax.scatter(z_paths[p, -1, 0], z_paths[p, -1, 1], s=30, c='red',   marker='x')
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title("Latent Space Paths")

ax = axes[1]
for p in range(min(50, curve_paths.shape[0])):
    ax.plot(tenors, curve_paths[p, -1, :] * 100, alpha=0.3, lw=0.8)
ax.plot(tenors, initial_curve * 100, 'k-', lw=2, label='Initial')
ax.plot(tenors, curve_paths[:, -1, :].mean(axis=0) * 100, 'r--', lw=2, label='Mean terminal')
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Swap Rate (%)")
ax.set_title("Terminal Curve Distribution"); ax.legend()

ax = axes[2]
tenor_idx = list(tenors).index(10)
time_axis = np.arange(curve_paths.shape[1]) / 12
for p in range(min(30, curve_paths.shape[0])):
    ax.plot(time_axis, curve_paths[p, :, tenor_idx] * 100, alpha=0.3, lw=0.8)
ax.plot(time_axis, curve_paths[:, :, tenor_idx].mean(axis=0) * 100, 'r-', lw=2, label='Mean')
ax.set_xlabel("Time (years)"); ax.set_ylabel("10Y Swap Rate (%)")
ax.set_title("10Y Rate Evolution"); ax.legend()

plt.suptitle("Forward Rate AEMM Simulation", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig_16_forward_aemm.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 18 - Multi-Factor Short Rate AEMM (Section 3.2.2)

# In[62]:


class ShortRateAEMM:
    """
    Multi-Factor Short Rate AEMM.

    Ornstein-Uhlenbeck dynamics on the VAE latent factors, with the full
    curve reconstructed by the decoder at each step.
    """

    def __init__(
        self,
        vae_model: VAE,
        tenors: np.ndarray,
        mean_reversion: np.ndarray,
        volatility: np.ndarray,
    ):
        self.vae = vae_model
        self.tenors = tenors
        self.kappa = mean_reversion
        self.sigma = volatility
        self.device = next(vae_model.parameters()).device

    def encode(self, swap_rates: np.ndarray) -> np.ndarray:
        rates_norm = normalize_rates(swap_rates).astype(np.float32)
        X = torch.from_numpy(rates_norm).to(self.device)
        return self.vae.get_latent(X)

    def decode(self, z: np.ndarray) -> np.ndarray:
        self.vae.eval()
        z_tensor = torch.from_numpy(z.astype(np.float32)).to(self.device)
        with torch.no_grad():
            decoded_norm = self.vae.decode(z_tensor).cpu().numpy()
        return denormalize_rates(decoded_norm)

    def get_short_rate(self, z: np.ndarray) -> float:
        """Linear extrapolation of the decoded curve to approximate the short rate."""
        curve = self.decode(z.reshape(1, -1))[0]
        short_rate = curve[0] - (curve[1] - curve[0]) * self.tenors[0] / (self.tenors[1] - self.tenors[0])
        return short_rate

    def simulate_paths(
        self,
        initial_curve: np.ndarray,
        n_paths: int,
        n_steps: int,
        dt: float,
        theta: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        dz_k = kappa_k * (theta_k - z_k) * dt + sigma_k * dW_k
        """
        latent_dim = self.vae.latent_dim
        n_tenors = len(self.tenors)

        z_init = self.encode(initial_curve.reshape(1, -1))[0]
        if theta is None:
            theta = z_init.copy()

        z_paths = np.zeros((n_paths, n_steps + 1, latent_dim))
        curve_paths = np.zeros((n_paths, n_steps + 1, n_tenors))
        short_rate_paths = np.zeros((n_paths, n_steps + 1))

        z_paths[:, 0, :] = z_init
        curve_paths[:, 0, :] = initial_curve
        short_rate_paths[:, 0] = self.get_short_rate(z_init)

        sqrt_dt = np.sqrt(dt)

        for t in range(n_steps):
            dW = np.random.randn(n_paths, latent_dim)
            for p in range(n_paths):
                z_current = z_paths[p, t, :]
                drift = self.kappa * (theta - z_current) * dt
                diffusion = self.sigma * dW[p] * sqrt_dt
                z_new = z_current + drift + diffusion
                curve_new = self.decode(z_new.reshape(1, -1))[0]

                z_paths[p, t + 1, :] = z_new
                curve_paths[p, t + 1, :] = curve_new
                short_rate_paths[p, t + 1] = self.get_short_rate(z_new)

        return z_paths, curve_paths, short_rate_paths


# In[63]:


print("\n" + "=" * 72)
print("Testing Short Rate AEMM")
print("=" * 72)

sr_aemm = ShortRateAEMM(
    vae_model=multi_vae,
    tenors=tenors,
    mean_reversion=np.array([0.1, 0.3]),
    volatility=np.array([0.005, 0.01]),
)

z_paths_sr, curve_paths_sr, short_rate_paths = sr_aemm.simulate_paths(
    initial_curve=initial_curve,
    n_paths=500,
    n_steps=60,
    dt=1/12,
)

print(f"\nSimulated {z_paths_sr.shape[0]} paths over {z_paths_sr.shape[1]-1} months")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
time_axis = np.arange(short_rate_paths.shape[1]) / 12
ax = axes[0]
for p in range(min(50, short_rate_paths.shape[0])):
    ax.plot(time_axis, short_rate_paths[p, :] * 100, alpha=0.2, lw=0.5)
ax.plot(time_axis, short_rate_paths.mean(axis=0) * 100, 'r-', lw=2, label='Mean')
ax.fill_between(
    time_axis,
    np.percentile(short_rate_paths, 5, axis=0) * 100,
    np.percentile(short_rate_paths, 95, axis=0) * 100,
    alpha=0.3, color='red', label='5-95% CI'
)
ax.set_xlabel("Time (years)"); ax.set_ylabel("Short Rate (%)")
ax.set_title("Short Rate Paths"); ax.legend()
ax = axes[1]
ax.hist(short_rate_paths[:, -1] * 100, bins=30, density=True, alpha=0.7, edgecolor='black')
ax.axvline(short_rate_paths[:, 0].mean() * 100,  color='green', linestyle='--', label='Initial')
ax.axvline(short_rate_paths[:, -1].mean() * 100, color='red',   linestyle='--', label='Mean terminal')
ax.set_xlabel("Short Rate (%)"); ax.set_ylabel("Density")
ax.set_title("Terminal Short Rate Distribution"); ax.legend()
ax = axes[2]
ax.plot(z_paths_sr[:, :, 0].mean(axis=0), z_paths_sr[:, :, 1].mean(axis=0), 'b-', lw=2, label='Mean path')
ax.scatter(z_paths_sr[:, 0, 0].mean(),  z_paths_sr[:, 0, 1].mean(),  s=100, c='green', marker='o', label='Start', zorder=5)
ax.scatter(z_paths_sr[:, -1, 0].mean(), z_paths_sr[:, -1, 1].mean(), s=100, c='red',   marker='x', label='End',   zorder=5)
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
ax.set_title("Mean Latent Path (O-U dynamics)"); ax.legend()
plt.suptitle("Short Rate AEMM Simulation", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig_17_short_rate_aemm.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 19 - Autoregressive AEMM (Section 4.1.2)
# 

# Chapter 4: P-Measure AEMM Models
# 1. **Autoregressive AEMM** (Section 4.1.2) — replacement for Dynamic Nelson-Siegel
# 2. **VAR(1)** on latent factors

# In[64]:


class AutoregressiveAEMM:
    """
    Autoregressive AEMM for P-measure forecasting.

    Fits VAR(1) on VAE latent factors: z_{t+1} = c + A z_t + eps_t.
    """

    def __init__(self, vae_model: VAE, tenors: np.ndarray):
        self.vae = vae_model
        self.tenors = tenors
        self.device = next(vae_model.parameters()).device

        self.c = None
        self.A = None
        self.Sigma = None
        self.z_history = None

    def encode(self, swap_rates: np.ndarray) -> np.ndarray:
        rates_norm = normalize_rates(swap_rates).astype(np.float32)
        X = torch.from_numpy(rates_norm).to(self.device)
        return self.vae.get_latent(X)

    def decode(self, z: np.ndarray) -> np.ndarray:
        self.vae.eval()
        z_tensor = torch.from_numpy(z.astype(np.float32)).to(self.device)
        with torch.no_grad():
            decoded_norm = self.vae.decode(z_tensor).cpu().numpy()
        return denormalize_rates(decoded_norm)

    def fit(self, historical_curves: np.ndarray):
        self.z_history = self.encode(historical_curves)
        T, K = self.z_history.shape

        Z_lag  = self.z_history[:-1]
        Z_lead = self.z_history[1:]
        X = np.column_stack([np.ones(T - 1), Z_lag])

        beta, _, _, _ = np.linalg.lstsq(X, Z_lead, rcond=None)
        self.c = beta[0]
        self.A = beta[1:].T

        residuals = Z_lead - X @ beta
        self.Sigma = np.cov(residuals.T)
        if self.Sigma.ndim == 0:
            self.Sigma = np.array([[self.Sigma]])

        print(f"VAR(1) fitted on {T} observations")
        print(f"Intercept c: {self.c}")
        print(f"AR matrix A eigenvalues: {np.linalg.eigvals(self.A)}")

    def forecast(
        self,
        initial_curve: np.ndarray,
        n_steps: int,
        n_paths: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.A is not None, "Must call fit() first"

        K = self.A.shape[0]
        N = len(self.tenors)

        z_init = self.encode(initial_curve.reshape(1, -1))[0]
        L = np.linalg.cholesky(self.Sigma)

        z_forecasts = np.zeros((n_paths, n_steps + 1, K))
        curve_forecasts = np.zeros((n_paths, n_steps + 1, N))
        z_forecasts[:, 0, :] = z_init
        curve_forecasts[:, 0, :] = initial_curve

        for t in range(n_steps):
            eps = np.random.randn(n_paths, K) @ L.T
            z_forecasts[:, t + 1, :] = self.c + z_forecasts[:, t, :] @ self.A.T + eps
            curve_forecasts[:, t + 1, :] = self.decode(z_forecasts[:, t + 1, :])

        return z_forecasts, curve_forecasts

    def point_forecast(self, initial_curve: np.ndarray, n_steps: int) -> np.ndarray:
        assert self.A is not None, "Must call fit() first"
        z = self.encode(initial_curve.reshape(1, -1))[0]
        forecasts = [initial_curve]
        for _ in range(n_steps):
            z = self.c + self.A @ z
            forecasts.append(self.decode(z.reshape(1, -1))[0])
        return np.array(forecasts)


# In[65]:


print("\n" + "=" * 72)
print("Fitting Autoregressive AEMM")
print("=" * 72)

demo_ccy = CURRENCIES[0]
historical_curves = swap_aligned[demo_ccy].values

ar_aemm = AutoregressiveAEMM(multi_vae, tenors)
ar_aemm.fit(historical_curves)
initial_curve = historical_curves[-1]
n_forecast_steps = 12

z_fcst, curve_fcst = ar_aemm.forecast(initial_curve, n_forecast_steps, n_paths=1000)
point_fcst = ar_aemm.point_forecast(initial_curve, n_forecast_steps)

print(f"\nGenerated {curve_fcst.shape[0]} forecast paths for {curve_fcst.shape[1]-1} periods")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
tenor_idx = list(tenors).index(10)
time_axis = np.arange(n_forecast_steps + 1)
for pct, alpha in [(5, 0.2), (25, 0.3), (50, 0.5)]:
    lower = np.percentile(curve_fcst[:, :, tenor_idx], pct,       axis=0) * 100
    upper = np.percentile(curve_fcst[:, :, tenor_idx], 100 - pct, axis=0) * 100
    ax.fill_between(time_axis, lower, upper, alpha=alpha, color='blue')
ax.plot(time_axis, point_fcst[:, tenor_idx] * 100, 'r-', lw=2, label='Point forecast')
ax.set_xlabel("Months ahead"); ax.set_ylabel("10Y Swap Rate (%)")
ax.set_title(f"{demo_ccy} 10Y Rate Fan Chart"); ax.legend()

ax = axes[1]
for p in range(min(100, curve_fcst.shape[0])):
    ax.plot(tenors, curve_fcst[p, -1, :] * 100, alpha=0.1, lw=0.5, color='blue')
ax.plot(tenors, initial_curve * 100, 'g-',  lw=2, label='Initial')
ax.plot(tenors, point_fcst[-1] * 100, 'r--', lw=2, label='Point forecast')
ax.plot(tenors, curve_fcst[:, -1, :].mean(axis=0) * 100, 'b-', lw=2, label='Mean forecast')
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Swap Rate (%)")
ax.set_title("12-Month Forecast Distribution"); ax.legend()

ax = axes[2]
ax.scatter(ar_aemm.z_history[:, 0], ar_aemm.z_history[:, 1], s=3, alpha=0.3, label='Historical', color='gray')
for p in range(min(50, z_fcst.shape[0])):
    ax.plot(z_fcst[p, :, 0], z_fcst[p, :, 1], alpha=0.2, lw=0.5, color='blue')
ax.plot(z_fcst[:, :, 0].mean(axis=0), z_fcst[:, :, 1].mean(axis=0), 'r-', lw=2, label='Mean path')
ax.scatter(z_fcst[:, 0, 0].mean(), z_fcst[:, 0, 1].mean(), s=100, c='green', marker='o', zorder=5)
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
ax.set_title("Latent Factor Forecasts"); ax.legend()
plt.suptitle(f"Autoregressive AEMM Forecasts ({demo_ccy})", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig_18_ar_aemm.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 19b - Statistical Comparison & Final Findings
# 
# This section consolidates the comparison NS vs Multi-VAE vs CVAE with:
# - Per-currency Wilcoxon signed-rank tests (paired RMSE)
# - Global pooled Wilcoxon test
# - Summary table (mean / median / p95) by method × currency
# - Final replication verdict

# In[66]:


# Aggregate per-method means/medians 
ns_means    = {c: float(np.mean(rmse_results[f"NelsonSiegel_{c}"]))   for c in CURRENCIES}
mvae_means  = {c: float(np.mean(rmse_results[f"MultiVAE_{c}"]))       for c in CURRENCIES}
cvae_means  = {c: float(np.mean(rmse_results[f"CVAE_{c}"]))           for c in CURRENCIES}
svae_means  = {c: float(np.mean(rmse_results[f"SingleVAE_{c}"]))      for c in CURRENCIES}

# Build a clean comparison DataFrame 
rows = []
def _add(ccy, method, arr):
    arr = np.asarray(arr)
    rows.append({
        "Currency": ccy,
        "Method":   method,
        "N":        len(arr),
        "Mean(bp)": float(np.mean(arr)),
        "Median(bp)": float(np.median(arr)),
        "P95(bp)":  float(np.percentile(arr, 95)),
    })

for ccy in CURRENCIES:
    _add(ccy, "NS (3D)",          rmse_results[f"NelsonSiegel_{ccy}"])
    _add(ccy, "Single VAE (2D)",  rmse_results[f"SingleVAE_{ccy}"])
    _add(ccy, "Multi VAE (2D)",   rmse_results[f"MultiVAE_{ccy}"])
    _add(ccy, "CVAE (2D)",        rmse_results[f"CVAE_{ccy}"])

df_comparison = pd.DataFrame(rows).sort_values(["Currency", "Method"]).reset_index(drop=True)
print("RMSE comparison table (bp):")
print(df_comparison.to_string(index=False))


# In[67]:


# Wilcoxon signed-rank tests: NS vs Multi-VAE (per currency)
print("\n" + "=" * 72)
print("Wilcoxon Signed-Rank Test: Multi-VAE (2D) vs Nelson-Siegel (3D)")
print("H0: median paired RMSE difference = 0")
print("=" * 72)

wilcoxon_mvae = {}
for ccy in CURRENCIES:
    ns_arr   = np.asarray(rmse_results[f"NelsonSiegel_{ccy}"])
    mvae_arr = np.asarray(rmse_results[f"MultiVAE_{ccy}"])
    n = min(len(ns_arr), len(mvae_arr))
    diff = ns_arr[:n] - mvae_arr[:n]
    if np.allclose(diff, 0.0):
        print(f"  {ccy}: diffs ≈ 0, skipped")
        continue
    stat, p = stats.wilcoxon(diff)
    winner = "Multi-VAE" if diff.mean() > 0 else "NS"
    wilcoxon_mvae[ccy] = {"meanΔ": diff.mean(), "p": p, "winner": winner}
    print(f"  {ccy}: meanΔ={diff.mean():+7.2f} bp   p={p:.2e}   [{winner} better]")

# Wilcoxon: NS vs CVAE 
print("\n" + "=" * 72)
print("Wilcoxon Signed-Rank Test: CVAE (2D) vs Nelson-Siegel (3D)")
print("=" * 72)

wilcoxon_cvae = {}
for ccy in CURRENCIES:
    ns_arr   = np.asarray(rmse_results[f"NelsonSiegel_{ccy}"])
    cvae_arr = np.asarray(rmse_results[f"CVAE_{ccy}"])
    n = min(len(ns_arr), len(cvae_arr))
    diff = ns_arr[:n] - cvae_arr[:n]
    if np.allclose(diff, 0.0):
        print(f"  {ccy}: diffs ≈ 0, skipped")
        continue
    stat, p = stats.wilcoxon(diff)
    winner = "CVAE" if diff.mean() > 0 else "NS"
    wilcoxon_cvae[ccy] = {"meanΔ": diff.mean(), "p": p, "winner": winner}
    print(f"  {ccy}: meanΔ={diff.mean():+7.2f} bp   p={p:.2e}   [{winner} better]")


# In[68]:


# Pooled global Wilcoxon (all currencies stacked) 
diff_pool_mvae = np.concatenate([
    np.asarray(rmse_results[f"NelsonSiegel_{c}"]) - np.asarray(rmse_results[f"MultiVAE_{c}"])
    for c in CURRENCIES
])
diff_pool_cvae = np.concatenate([
    np.asarray(rmse_results[f"NelsonSiegel_{c}"]) - np.asarray(rmse_results[f"CVAE_{c}"])
    for c in CURRENCIES
])

print("\n" + "=" * 72)
print("Pooled Wilcoxon (all currencies)")
print("=" * 72)

stat_m, p_m = stats.wilcoxon(diff_pool_mvae)
print(f"  NS vs Multi-VAE: meanΔ={diff_pool_mvae.mean():+.2f} bp   p={p_m:.2e}   "
      f"[{'Multi-VAE' if diff_pool_mvae.mean() > 0 else 'NS'} better]")

stat_c, p_c = stats.wilcoxon(diff_pool_cvae)
print(f"  NS vs CVAE     : meanΔ={diff_pool_cvae.mean():+.2f} bp   p={p_c:.2e}   "
      f"[{'CVAE' if diff_pool_cvae.mean() > 0 else 'NS'} better]")


# In[69]:


# Final replication verdict 
sep = "=" * 72

def _fmt_range(d):
    v = list(d.values())
    return f"{min(v):.2f}–{max(v):.2f} bp"

print("\n" + sep)
print("  REPLICATION SUMMARY — Sokol (2022) AEMM")
print(sep)

print(f"\n  Sample: {len(CURRENCIES)} currencies × {len(TARGET_TENORS)} tenors, "
      f"{X_multi.shape[0]} total observations")
print(f"  Train cutoff: {TRAIN_CUTOFF.strftime('%Y-%m-%d')}")

print("\n  Mean RMSE ranges (across currencies):")
print(f"    Nelson-Siegel (3D)  : {_fmt_range(ns_means)}")
print(f"    Single VAE    (2D)  : {_fmt_range(svae_means)}")
print(f"    Multi VAE     (2D)  : {_fmt_range(mvae_means)}")
print(f"    CVAE          (2D)  : {_fmt_range(cvae_means)}")

multi_beats_ns = sum(1 for c in CURRENCIES if mvae_means[c] < ns_means[c])
cvae_beats_ns  = sum(1 for c in CURRENCIES if cvae_means[c] < ns_means[c])

print(f"\n  Multi-VAE beats NS: {multi_beats_ns}/{len(CURRENCIES)} currencies")
print(f"  CVAE      beats NS: {cvae_beats_ns}/{len(CURRENCIES)} currencies")

# OOS sanity
if len(rmse_oos) > 0:
    ratio = rmse_oos.mean() / rmse_insample.mean()
    print(f"\n  Out-of-sample sanity:")
    print(f"    In-sample  mean RMSE : {rmse_insample.mean():.2f} bp")
    print(f"    Out-sample mean RMSE : {rmse_oos.mean():.2f} bp")
    print(f"    Ratio OOS/IS         : {ratio:.2f}×  "
          f"({'no overfitting' if ratio < 1.5 else 'possible overfitting'})")

print("\n  Key findings:")
print(f"    1. Multi-VAE (2D) achieves comparable accuracy to NS (3D) "
      f"with one fewer latent dimension.")
print(f"    2. The shared latent space lets one model handle all "
      f"{len(CURRENCIES)} currencies simultaneously.")
print(f"    3. Latent dimensions z1/z2 separate currencies by rate regime "
      f"(see Figure 13).")
print(sep)


# 
# ## 20 - Save All Results

# In[70]:


vae_output = {
    "single_vae_models": {ccy: m.state_dict() for ccy, m in single_vae_models.items()},
    "multi_vae_state": multi_vae.state_dict(),
    "cvae_state": cvae.state_dict(),
    "rmse_results": rmse_results,
    "multi_vae_history": multi_vae_history,
    "cvae_history": cvae_history,
    "ns_results": ns_results,
    "config": {
        "S_MIN": S_MIN,
        "S_MAX": S_MAX,
        "rates_units": "decimal",
        "bp_per_unit": BP_PER_UNIT,
        "latent_dim": 2,
        "input_dim": 7,
        "beta": 1e-7,
        "currencies": CURRENCIES,
        "target_tenors": TARGET_TENORS,
        "seed": SEED,
        "train_cutoff": str(TRAIN_CUTOFF),
         "df_comparison": df_comparison,
        "wilcoxon_mvae": wilcoxon_mvae,
        "wilcoxon_cvae": wilcoxon_cvae,
    },
}

with open("results/vae_results_extended.pkl", "wb") as f:
    pickle.dump(vae_output, f)

torch.save(multi_vae.state_dict(), "results/multi_vae_weights.pt")
torch.save(cvae.state_dict(),      "results/cvae_weights.pt")

