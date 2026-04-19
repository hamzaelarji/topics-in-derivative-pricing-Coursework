# Notebook 3 - VAE Training 

This notebook implements and trains the VAE architectures described in Sokol (2022), "Autoencoder Market Models for Interest Rates".
1. **Single-Currency VAE** : trained on one currency at a time
2. **Multi-Currency VAE** : trained on all currencies with shared latent space
3. **Multi-Currency CVAE** : conditional VAE with one-hot currency encoding

### Architecture (Tables 1-3 in the paper):
- **Input**: N=7 swap rates (2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y)
- **Latent space**: K=2 dimensions
- **Pre-processing**: linear map from [-5%, 25%] to [0, 1]
- **Loss**: L2 reconstruction + β·KLD (β = 1e-7)

### Extensions:
- **Post-processing gradient descent** (Section 2.3.1)
- **Nelson-Siegel comparison** (Section 2.4.3)
- **In-sample vs Out-of-sample validation** (Figure 12)
- **Paper figures replication** (Figures 9, 10, 11, 12, 13, 14, 15)
- **Chapter 3: Q-Measure AEMM Models** (Forward Rate & Short Rate)
- **Chapter 4: P-Measure AEMM Models** (Autoregressive)

## 0 - Imports


```python
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
```


```python
Path("figures").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
```

## 1 - Configuration & Data


```python
SEED = 42  # reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

    Using device: cpu



```python
DATA_PATH = Path("data/df_multi.csv")
EXCLUDE_CURRENCIES = {"CHF"} 
df_long = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df_long = df_long[~df_long["currency"].isin(EXCLUDE_CURRENCIES)].copy()
```


```python
CURRENCIES = sorted(df_long["currency"].unique())
TARGET_TENORS = [2, 3, 5, 10, 15, 20, 30]
TENORS = np.array(TARGET_TENORS, dtype=float)
TENOR_COLS = [str(t) for t in TARGET_TENORS]    # column names in the CSV are strings
```


```python
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
```


```python
swap_data = swap_aligned
```


```python
dates_ref = swap_aligned[CURRENCIES[0]].index
print(f"Loaded {len(CURRENCIES)} currencies: {CURRENCIES}")
print(f"Dates: {len(dates_ref)}  "
      f"({dates_ref.min().date()} -> {dates_ref.max().date()})")
print(f"Tenors: {TARGET_TENORS}")
for ccy in CURRENCIES:
    print(f"  {ccy}: shape={swap_aligned[ccy].shape}")
```

    Loaded 7 currencies: ['AUD', 'CAD', 'DKK', 'EUR', 'GBP', 'JPY', 'USD']
    Dates: 774  (2023-01-30 -> 2026-01-15)
    Tenors: [2, 3, 5, 10, 15, 20, 30]
      AUD: shape=(774, 7)
      CAD: shape=(774, 7)
      DKK: shape=(774, 7)
      EUR: shape=(774, 7)
      GBP: shape=(774, 7)
      JPY: shape=(774, 7)
      USD: shape=(774, 7)



```python
# Use all available currencies in swap_aligned
CURRENCIES = list(swap_aligned.keys())
N_CCY = len(CURRENCIES)
CCY_TO_IDX = {c: i for i, c in enumerate(CURRENCIES)}
print("Dataset used: swap_aligned")
for ccy, df in swap_aligned.items():
    print(f"  {ccy}: {df.shape[0]} obs × {df.shape[1]} tenors")
```

    Dataset used: swap_aligned
      AUD: 774 obs × 7 tenors
      CAD: 774 obs × 7 tenors
      DKK: 774 obs × 7 tenors
      EUR: 774 obs × 7 tenors
      GBP: 774 obs × 7 tenors
      JPY: 774 obs × 7 tenors
      USD: 774 obs × 7 tenors


## 2 - Data pre-processing


```python
# Following the paper exactly:
# - Map swap rates from [S_min, S_max] = [-5%, 25%] to [0, 1] using linear transform
# - This matches the Sigmoid output activation of the decoder

S_MIN = -0.05   # lower bound (decimal)
S_MAX = 0.25    # upper bound (decimal)
BP_PER_UNIT = 10000.0
```


```python
def normalize_rates(rates: np.ndarray) -> np.ndarray:
    """Map swap rates (decimal) from [S_MIN, S_MAX] to [0, 1]."""
    x = (rates - S_MIN) / (S_MAX - S_MIN)
    return np.clip(x, 0.0, 1.0)
```


```python
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
```


```python
multi_rates = np.vstack(multi_rates)
multi_labels = np.vstack(multi_labels)
```


```python
X_multi = torch.from_numpy(multi_rates)
Y_multi = torch.from_numpy(multi_labels)
```


```python
print(f"\nMulti-currency dataset: {X_multi.shape[0]} samples × {X_multi.shape[1]} features")
print(f"Currency distribution: {pd.Series(multi_ccy_ids).value_counts().to_dict()}")
## 2b - Train/Test Split for Out-of-Sample Validation
```

    
    Multi-currency dataset: 5418 samples × 7 features
    Currency distribution: {'AUD': 774, 'CAD': 774, 'DKK': 774, 'EUR': 774, 'GBP': 774, 'JPY': 774, 'USD': 774}



```python
# Following the paper (Section 2.4.2). Adjust `TRAIN_CUTOFF` to match your data range.
# TRAIN_CUTOFF = pd.Timestamp("2025-07-31") # 80 train / 20 tests
# TRAIN_CUTOFF = pd.Timestamp("2024-07-31") # 50 train / 50 tests
TRAIN_CUTOFF = pd.Timestamp("2025-03-31") # split 70/30 (recommandée pour ton cas)
```


```python
def create_train_test_split(swap_aligned: Dict, cutoff_date: pd.Timestamp):
    """Split aligned swap data into train (<=cutoff) and test (>cutoff)."""
    train_data, test_data = {}, {}
    for ccy, df in swap_aligned.items():
        train_data[ccy] = df[df.index <= cutoff_date]
        test_data[ccy]  = df[df.index >  cutoff_date]
    return train_data, test_data
```


```python
swap_aligned_train, swap_aligned_test = create_train_test_split(swap_aligned, TRAIN_CUTOFF)
```


```python
print(f"Train/Test split at {TRAIN_CUTOFF.strftime('%Y-%m-%d')}")
print("\nTraining set:")
for ccy, df in swap_aligned_train.items():
    if len(df) > 0:
        print(f"  {ccy}: {len(df)} obs ({df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')})")
    else:
        print(f"  {ccy}: 0 observations")
```

    Train/Test split at 2025-03-31
    
    Training set:
      AUD: 566 obs (2023-01 to 2025-03)
      CAD: 566 obs (2023-01 to 2025-03)
      DKK: 566 obs (2023-01 to 2025-03)
      EUR: 566 obs (2023-01 to 2025-03)
      GBP: 566 obs (2023-01 to 2025-03)
      JPY: 566 obs (2023-01 to 2025-03)
      USD: 566 obs (2023-01 to 2025-03)



```python
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
```

    
    Test set:
      AUD: 208 obs (2025-04 to 2026-01)
      CAD: 208 obs (2025-04 to 2026-01)
      DKK: 208 obs (2025-04 to 2026-01)
      EUR: 208 obs (2025-04 to 2026-01)
      GBP: 208 obs (2025-04 to 2026-01)
      JPY: 208 obs (2025-04 to 2026-01)
      USD: 208 obs (2025-04 to 2026-01)



```python
X_train, Y_train, train_ccy_ids, train_dates = prepare_multi_currency_tensors(swap_aligned_train, CURRENCIES)
if sum(len(df) for df in swap_aligned_test.values()) > 0:
    X_test, Y_test, test_ccy_ids, test_dates = prepare_multi_currency_tensors(swap_aligned_test, CURRENCIES)
else:
    X_test = torch.empty(0, 7)
    Y_test = torch.empty(0, N_CCY)
    test_ccy_ids, test_dates = [], []
```


```python
print(f"\nTraining tensors: X_train {tuple(X_train.shape)}, Y_train {tuple(Y_train.shape)}")
print(f"Test tensors:     X_test  {tuple(X_test.shape)},  Y_test  {tuple(Y_test.shape)}")
```

    
    Training tensors: X_train (3962, 7), Y_train (3962, 7)
    Test tensors:     X_test  (1456, 7),  Y_test  (1456, 7)


## 3 - VAE Architecture


```python
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

```

    
     Model classes defined
       VAE single-ccy params: 99
       VAE multi-ccy params:  191
       CVAE params:           433


## 4 - Loss function (FIXED)

From Eq. (5) of the paper:
$$\mathcal{L}_{VAE} = \frac{1}{|\text{batch}|}\sum_{i \in \text{batch}} \sum_{n=1}^{N}(S_{i,n}-S'_{i,n})^2 + \beta \cdot D_{KLD}(\mu, \sigma)$$


```python
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
```

## 5 - Training loop


```python
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
```

## 6 - Train all three architectures


```python
print("\n" + "=" * 72)
print("Training SINGLE-CURRENCY VAEs")
print("=" * 72)
single_vae_models: Dict[str, VAE] = {}
single_vae_histories: Dict[str, Dict] = {}

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

```

    
    ========================================================================
    Training SINGLE-CURRENCY VAEs
    ========================================================================
    
    ──────────────────────────────────────────────────
      AUD
    ──────────────────────────────────────────────────
      Epoch     1/5000 | Loss: 5.387342e-01 | Recon: 5.387342e-01 | KLD: 7.822e-02 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 6.795287e-04 | Recon: 6.788450e-04 | KLD: 6.837e+00 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 7.169996e-04 | Recon: 7.160596e-04 | KLD: 9.400e+00 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 6.672277e-04 | Recon: 6.661752e-04 | KLD: 1.052e+01 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 1.748342e-04 | Recon: 1.738950e-04 | KLD: 9.391e+00 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 1.758047e-04 | Recon: 1.747703e-04 | KLD: 1.034e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 1.633649e-04 | Recon: 1.623196e-04 | KLD: 1.045e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 1.583287e-04 | Recon: 1.572901e-04 | KLD: 1.039e+01 | LR: 5.0e-04
      Epoch  4000/5000 | Loss: 1.735682e-04 | Recon: 1.725411e-04 | KLD: 1.027e+01 | LR: 5.0e-04
      Epoch  4500/5000 | Loss: 1.605214e-04 | Recon: 1.594998e-04 | KLD: 1.022e+01 | LR: 2.5e-04
      Epoch  5000/5000 | Loss: 1.521977e-04 | Recon: 1.511771e-04 | KLD: 1.021e+01 | LR: 1.3e-04
    
    ──────────────────────────────────────────────────
      CAD
    ──────────────────────────────────────────────────
      Epoch     1/5000 | Loss: 4.924021e-01 | Recon: 4.924021e-01 | KLD: 5.961e-02 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 2.134142e-03 | Recon: 2.133590e-03 | KLD: 5.512e+00 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 2.312949e-03 | Recon: 2.312249e-03 | KLD: 6.993e+00 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 3.006192e-04 | Recon: 2.997602e-04 | KLD: 8.590e+00 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 1.673740e-04 | Recon: 1.662980e-04 | KLD: 1.076e+01 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 1.619636e-04 | Recon: 1.608039e-04 | KLD: 1.160e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 1.473267e-04 | Recon: 1.460269e-04 | KLD: 1.300e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 1.535477e-04 | Recon: 1.522014e-04 | KLD: 1.346e+01 | LR: 1.0e-03
      Epoch  4000/5000 | Loss: 1.481780e-04 | Recon: 1.468313e-04 | KLD: 1.347e+01 | LR: 1.0e-03
      Epoch  4500/5000 | Loss: 1.408434e-04 | Recon: 1.394968e-04 | KLD: 1.347e+01 | LR: 1.0e-03
      Epoch  5000/5000 | Loss: 1.444681e-04 | Recon: 1.431505e-04 | KLD: 1.318e+01 | LR: 1.0e-03
    
    ──────────────────────────────────────────────────
      DKK
    ──────────────────────────────────────────────────
      Epoch     1/5000 | Loss: 4.828734e-01 | Recon: 4.828734e-01 | KLD: 1.269e-01 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 1.345813e-03 | Recon: 1.345392e-03 | KLD: 4.211e+00 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 1.312098e-03 | Recon: 1.311478e-03 | KLD: 6.202e+00 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 2.672459e-04 | Recon: 2.665158e-04 | KLD: 7.301e+00 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 2.778552e-04 | Recon: 2.768910e-04 | KLD: 9.642e+00 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 2.347523e-04 | Recon: 2.337049e-04 | KLD: 1.047e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 2.300396e-04 | Recon: 2.289684e-04 | KLD: 1.071e+01 | LR: 5.0e-04
      Epoch  3500/5000 | Loss: 2.401014e-04 | Recon: 2.390433e-04 | KLD: 1.058e+01 | LR: 5.0e-04
      Epoch  4000/5000 | Loss: 2.586397e-04 | Recon: 2.575882e-04 | KLD: 1.052e+01 | LR: 2.5e-04
      Epoch  4500/5000 | Loss: 2.798342e-04 | Recon: 2.788043e-04 | KLD: 1.030e+01 | LR: 1.3e-04
      Epoch  5000/5000 | Loss: 2.321421e-04 | Recon: 2.311052e-04 | KLD: 1.037e+01 | LR: 6.3e-05
    
    ──────────────────────────────────────────────────
      EUR
    ──────────────────────────────────────────────────
      Epoch     1/5000 | Loss: 4.001455e-01 | Recon: 4.001455e-01 | KLD: 2.994e-02 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 1.209576e-03 | Recon: 1.209157e-03 | KLD: 4.193e+00 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 1.284696e-03 | Recon: 1.284164e-03 | KLD: 5.321e+00 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 3.782657e-04 | Recon: 3.775718e-04 | KLD: 6.939e+00 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 1.855268e-04 | Recon: 1.846686e-04 | KLD: 8.582e+00 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 1.753461e-04 | Recon: 1.744304e-04 | KLD: 9.157e+00 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 2.085773e-04 | Recon: 2.076300e-04 | KLD: 9.472e+00 | LR: 5.0e-04
      Epoch  3500/5000 | Loss: 1.842261e-04 | Recon: 1.832629e-04 | KLD: 9.632e+00 | LR: 5.0e-04
      Epoch  4000/5000 | Loss: 1.810502e-04 | Recon: 1.800893e-04 | KLD: 9.610e+00 | LR: 2.5e-04
      Epoch  4500/5000 | Loss: 1.982373e-04 | Recon: 1.972854e-04 | KLD: 9.518e+00 | LR: 2.5e-04
      Epoch  5000/5000 | Loss: 1.761564e-04 | Recon: 1.752068e-04 | KLD: 9.496e+00 | LR: 1.3e-04
    
    ──────────────────────────────────────────────────
      GBP
    ──────────────────────────────────────────────────
      Epoch     1/5000 | Loss: 4.291355e-01 | Recon: 4.291355e-01 | KLD: 9.040e-02 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 1.379765e-03 | Recon: 1.379236e-03 | KLD: 5.285e+00 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 1.361837e-03 | Recon: 1.361099e-03 | KLD: 7.386e+00 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 1.371799e-03 | Recon: 1.370751e-03 | KLD: 1.048e+01 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 4.966739e-04 | Recon: 4.955181e-04 | KLD: 1.156e+01 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 5.056529e-04 | Recon: 5.045149e-04 | KLD: 1.138e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 2.606655e-04 | Recon: 2.596269e-04 | KLD: 1.039e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 1.494888e-05 | Recon: 1.368279e-05 | KLD: 1.266e+01 | LR: 1.0e-03
      Epoch  4000/5000 | Loss: 9.837914e-06 | Recon: 8.377860e-06 | KLD: 1.460e+01 | LR: 1.0e-03
      Epoch  4500/5000 | Loss: 9.382355e-06 | Recon: 7.887704e-06 | KLD: 1.495e+01 | LR: 1.0e-03
      Epoch  5000/5000 | Loss: 7.733140e-06 | Recon: 6.257391e-06 | KLD: 1.476e+01 | LR: 1.0e-03
    
    ──────────────────────────────────────────────────
      JPY
    ──────────────────────────────────────────────────
      Epoch     1/5000 | Loss: 6.134513e-01 | Recon: 6.134513e-01 | KLD: 2.089e-01 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 1.096417e-03 | Recon: 1.095499e-03 | KLD: 9.175e+00 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 9.842211e-04 | Recon: 9.829690e-04 | KLD: 1.252e+01 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 1.088609e-03 | Recon: 1.087344e-03 | KLD: 1.265e+01 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 2.939657e-05 | Recon: 2.823748e-05 | KLD: 1.159e+01 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 2.576235e-05 | Recon: 2.444913e-05 | KLD: 1.313e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 2.528889e-05 | Recon: 2.389043e-05 | KLD: 1.398e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 2.742786e-05 | Recon: 2.603151e-05 | KLD: 1.396e+01 | LR: 1.0e-03
      Epoch  4000/5000 | Loss: 2.771041e-05 | Recon: 2.632243e-05 | KLD: 1.388e+01 | LR: 1.0e-03
      Epoch  4500/5000 | Loss: 2.745865e-05 | Recon: 2.607579e-05 | KLD: 1.383e+01 | LR: 5.0e-04
      Epoch  5000/5000 | Loss: 2.626642e-05 | Recon: 2.491186e-05 | KLD: 1.355e+01 | LR: 5.0e-04
    
    ──────────────────────────────────────────────────
      USD
    ──────────────────────────────────────────────────
      Epoch     1/5000 | Loss: 2.610236e-01 | Recon: 2.610236e-01 | KLD: 7.324e-02 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 1.141557e-03 | Recon: 1.141063e-03 | KLD: 4.942e+00 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 1.085739e-03 | Recon: 1.085090e-03 | KLD: 6.485e+00 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 1.017750e-03 | Recon: 1.016790e-03 | KLD: 9.597e+00 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 4.406576e-04 | Recon: 4.396399e-04 | KLD: 1.018e+01 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 4.352955e-04 | Recon: 4.343118e-04 | KLD: 9.837e+00 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 4.371916e-04 | Recon: 4.361771e-04 | KLD: 1.014e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 4.553014e-04 | Recon: 4.542752e-04 | KLD: 1.026e+01 | LR: 5.0e-04
      Epoch  4000/5000 | Loss: 5.095084e-04 | Recon: 5.084735e-04 | KLD: 1.035e+01 | LR: 5.0e-04
      Epoch  4500/5000 | Loss: 3.855667e-04 | Recon: 3.844936e-04 | KLD: 1.073e+01 | LR: 2.5e-04
      Epoch  5000/5000 | Loss: 9.080161e-05 | Recon: 8.960416e-05 | KLD: 1.197e+01 | LR: 2.5e-04
    
    ========================================================================
    Training MULTI-CURRENCY VAE (shared latent space)
    ========================================================================
      Epoch     1/5000 | Loss: 3.039189e-01 | Recon: 3.039188e-01 | KLD: 1.629e-01 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 4.910044e-04 | Recon: 4.898881e-04 | KLD: 1.116e+01 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 4.826409e-04 | Recon: 4.816573e-04 | KLD: 9.836e+00 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 3.748048e-05 | Recon: 3.589342e-05 | KLD: 1.587e+01 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 3.491211e-05 | Recon: 3.334494e-05 | KLD: 1.567e+01 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 3.434402e-05 | Recon: 3.277963e-05 | KLD: 1.564e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 3.354596e-05 | Recon: 3.198673e-05 | KLD: 1.559e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 3.375840e-05 | Recon: 3.221455e-05 | KLD: 1.544e+01 | LR: 1.0e-03
      Epoch  4000/5000 | Loss: 3.390123e-05 | Recon: 3.236936e-05 | KLD: 1.532e+01 | LR: 1.0e-03
      Epoch  4500/5000 | Loss: 3.258076e-05 | Recon: 3.106033e-05 | KLD: 1.520e+01 | LR: 1.0e-03
      Epoch  5000/5000 | Loss: 3.276580e-05 | Recon: 3.125363e-05 | KLD: 1.512e+01 | LR: 1.0e-03
    
    ========================================================================
    Training MULTI-CURRENCY CVAE (conditional)
    ========================================================================
      Epoch     1/5000 | Loss: 3.671580e-01 | Recon: 3.671580e-01 | KLD: 1.110e-01 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 2.583067e-05 | Recon: 2.453981e-05 | KLD: 1.291e+01 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 2.131444e-05 | Recon: 1.992417e-05 | KLD: 1.390e+01 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 1.952965e-05 | Recon: 1.816853e-05 | KLD: 1.361e+01 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 1.876318e-05 | Recon: 1.741811e-05 | KLD: 1.345e+01 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 1.863993e-05 | Recon: 1.730333e-05 | KLD: 1.337e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 1.896390e-05 | Recon: 1.763875e-05 | KLD: 1.325e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 1.887280e-05 | Recon: 1.755201e-05 | KLD: 1.321e+01 | LR: 1.0e-03
      Epoch  4000/5000 | Loss: 1.826595e-05 | Recon: 1.695378e-05 | KLD: 1.312e+01 | LR: 1.0e-03
      Epoch  4500/5000 | Loss: 1.853468e-05 | Recon: 1.722969e-05 | KLD: 1.305e+01 | LR: 1.0e-03
      Epoch  5000/5000 | Loss: 1.791704e-05 | Recon: 1.661704e-05 | KLD: 1.300e+01 | LR: 1.0e-03


### 6b - Post-Processing Gradient Descent (Section 2.3.1)

From the paper:
> "A post-processing step increases the accuracy of VAE mapping by performing gradient
> descent minimizing L2 loss starting from the center μ(S) of the distribution produced
> by the encoder."


```python
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
```

    
    ========================================================================
    Applying Post-Processing Gradient Descent
    ========================================================================



```python
print("\nMulti-Currency VAE:")
z_refined_multi, x_recon_refined_multi = post_processing_gradient_descent(
    multi_vae, X_multi, n_steps=100, lr=0.01, verbose=True
)
```

    
    Multi-Currency VAE:
      Step 20/100: L2 loss = 3.115075e-05
      Step 40/100: L2 loss = 2.989187e-05
      Step 60/100: L2 loss = 2.972345e-05
      Step 80/100: L2 loss = 2.970702e-05
      Step 100/100: L2 loss = 2.970432e-05



```python
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
```

    
    CVAE:
      Step 20/100: L2 loss = 1.600720e-05
      Step 40/100: L2 loss = 1.596226e-05
      Step 60/100: L2 loss = 1.595741e-05
      Step 80/100: L2 loss = 1.595656e-05
      Step 100/100: L2 loss = 1.595646e-05



```python
multi_vae.eval()
with torch.no_grad():
    x_recon_before, _, _ = multi_vae(X_multi.to(device))
    x_recon_before = x_recon_before.cpu().numpy()
```


```python
rmse_before = compute_rmse_from_recon(x_recon_before, X_multi.numpy())
rmse_after  = compute_rmse_from_recon(x_recon_refined_multi, X_multi.numpy())
```


```python
print("\nMulti-Currency VAE RMSE Comparison:")
print(f"  Before post-processing: Mean={np.mean(rmse_before):.2f} bp, Median={np.median(rmse_before):.2f} bp")
print(f"  After  post-processing: Mean={np.mean(rmse_after):.2f} bp, Median={np.median(rmse_after):.2f} bp")
print(f"  Improvement: {(1 - np.mean(rmse_after)/np.mean(rmse_before))*100:.1f}%")

```

    
    Multi-Currency VAE RMSE Comparison:
      Before post-processing: Mean=5.49 bp, Median=5.04 bp
      After  post-processing: Mean=5.35 bp, Median=4.87 bp
      Improvement: 2.5%


The modest improvement (2.5%) reflects that with β=1e-7, the KLD penalty barely biases μ(S) away from the L2 optimum, so post-processing has little room to improve.

The reported improvement of 2.5% is computed on the full dataset (X_multi), i.e. in-sample. It therefore reflects the overall effect of the post-processing rather than out-of-sample predictive performance.

## 7 - Nelson-Siegel Implementation (Section 2.2)

Classical Nelson-Siegel basis with 3 factors:
$$S(\tau) = \beta_1 + \beta_2 \frac{1-e^{-\tau/\lambda}}{\tau/\lambda} + \beta_3 \left( \frac{1-e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda} \right)$$



```python
with open("results/ns_results.pkl", "rb") as f:
    ns_nb2 = pickle.load(f)
    
LAM_STAR = ns_nb2["lambda_star"]  # = 0.5
print(LAM_STAR)
```

    0.49999999999999994



```python
class NelsonSiegel:
    """Nelson-Siegel curve fitting with fixed lambda."""

    def __init__(self, lambda_param: float = LAM_STAR):
        self.lambda_param = lambda_param

    def basis_functions(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        lam = self.lambda_param
        f1 = np.ones_like(tau)
        lt = lam * tau
        with np.errstate(divide="ignore", invalid="ignore"):
            f2 = np.where(lt > 1e-10, (1.0 - np.exp(-lt)) / lt, 1.0)
        f3 = f2 - np.exp(-lt)
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
```


```python
print("\n" + "=" * 72)
print("Fitting Nelson-Siegel Model")
print("=" * 72)
ns_model = NelsonSiegel(lambda_param=LAM_STAR)  # λ★ from NB2 pooled cross-validation
tenors = np.array(TARGET_TENORS, dtype=float)
ns_results: Dict[str, Dict] = {}
for ccy in CURRENCIES:
    rates = swap_aligned[ccy].values
    betas, fitted = ns_model.fit_multiple(tenors, rates)
    rmse_bp = np.sqrt(np.mean((fitted - rates) ** 2, axis=1)) * BP_PER_UNIT
    ns_results[ccy] = {"betas": betas, "fitted": fitted, "rmse_bp": rmse_bp}
    print(f"  {ccy}: Mean RMSE = {np.mean(rmse_bp):.2f} bp, Median = {np.median(rmse_bp):.2f} bp")
```

    
    ========================================================================
    Fitting Nelson-Siegel Model
    ========================================================================
      AUD: Mean RMSE = 10.00 bp, Median = 9.47 bp
      CAD: Mean RMSE = 7.43 bp, Median = 7.00 bp
      DKK: Mean RMSE = 8.31 bp, Median = 8.02 bp
      EUR: Mean RMSE = 6.35 bp, Median = 6.50 bp
      GBP: Mean RMSE = 2.53 bp, Median = 2.48 bp
      JPY: Mean RMSE = 6.49 bp, Median = 6.10 bp
      USD: Mean RMSE = 6.28 bp, Median = 5.70 bp


## 8 - Plot training convergence


```python
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
```


    
![png](03_vae_training_files/03_vae_training_52_0.png)
    


## 9 - Compute reconstruction RMSE for all models

Single-VAE is now evaluated only on the aligned dates so it is directly comparable
to Multi-VAE / CVAE / Nelson-Siegel (which all live on `swap_aligned`).


```python
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
```


```python
# Single-currency VAE — evaluate only on aligned dates for a fair comparison
for ccy in CURRENCIES:
    aligned_dates = swap_aligned[ccy].index
    full_dates    = single_ccy_data[ccy]["dates"]
    mask = full_dates.isin(aligned_dates)
    X_aligned = single_ccy_data[ccy]["tensor"][mask]
    rmse_bp, _ = compute_rmse_bp(single_vae_models[ccy], X_aligned)
    rmse_results[f"SingleVAE_{ccy}"] = rmse_bp
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 2
          1 # Single-currency VAE — evaluate only on aligned dates for a fair comparison
    ----> 2 for ccy in CURRENCIES:
          3     aligned_dates = swap_aligned[ccy].index
          4     full_dates    = single_ccy_data[ccy]["dates"]


    NameError: name 'CURRENCIES' is not defined



```python
# Multi-currency VAE (per-currency slices)
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    X_slice = X_multi[offset:offset+n]
    rmse_bp, _ = compute_rmse_bp(multi_vae, X_slice)
    rmse_results[f"MultiVAE_{ccy}"] = rmse_bp
    offset += n
```


```python
# CVAE (per-currency slices)
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    X_slice = X_multi[offset:offset+n]
    Y_slice = Y_multi[offset:offset+n]
    rmse_bp, _ = compute_rmse_bp(cvae, X_slice, Y_slice)
    rmse_results[f"CVAE_{ccy}"] = rmse_bp
    offset += n
```


```python
# Nelson-Siegel
for ccy in CURRENCIES:
    rmse_results[f"NelsonSiegel_{ccy}"] = ns_results[ccy]["rmse_bp"]
```


```python
print(f"\n{'Model':<20} {'Currency':<8} {'Mean(bp)':>10} {'Median':>10} {'95th':>10}")
print("─" * 64)
for key, rmse in rmse_results.items():
    model_name, ccy = key.rsplit("_", 1)
    print(f"{model_name:<20} {ccy:<8} {np.mean(rmse):>10.2f} {np.median(rmse):>10.2f} {np.percentile(rmse, 95):>10.2f}")
```

    
    Model                Currency   Mean(bp)     Median       95th
    ────────────────────────────────────────────────────────────────
    SingleVAE            AUD           12.97      12.58      23.72
    SingleVAE            CAD           11.85      10.13      24.16
    SingleVAE            DKK           15.75      14.41      31.27
    SingleVAE            EUR           13.20      11.85      28.93
    SingleVAE            GBP            2.48       1.97       5.64
    SingleVAE            JPY            4.99       4.71       9.88
    SingleVAE            USD            8.11       6.98      15.70
    MultiVAE             AUD            5.77       5.00      11.46
    MultiVAE             CAD            6.10       5.44      13.35
    MultiVAE             DKK            6.28       5.74      13.08
    MultiVAE             EUR            3.38       3.06       6.22
    MultiVAE             GBP            6.78       6.64       9.90
    MultiVAE             JPY            4.63       4.53       8.17
    MultiVAE             USD            5.52       4.59      11.83
    CVAE                 AUD            4.00       3.39       8.67
    CVAE                 CAD            4.53       4.17       9.13
    CVAE                 DKK            4.88       4.13      11.45
    CVAE                 EUR            2.75       2.43       5.51
    CVAE                 GBP            2.49       2.03       5.83
    CVAE                 JPY            3.00       2.71       5.66
    CVAE                 USD            4.64       3.77      10.13
    NelsonSiegel         AUD           10.00       9.47      16.23
    NelsonSiegel         CAD            7.43       7.00      12.05
    NelsonSiegel         DKK            8.31       8.02      15.53
    NelsonSiegel         EUR            6.35       6.50       8.80
    NelsonSiegel         GBP            2.53       2.48       3.19
    NelsonSiegel         JPY            6.49       6.10      11.66
    NelsonSiegel         USD            6.28       5.70      10.80


## 10 - Figure 9: RMSE Distribution by Model Type


```python
rmse_single_all = np.concatenate([rmse_results[f"SingleVAE_{ccy}"] for ccy in CURRENCIES])
rmse_multi_all  = np.concatenate([rmse_results[f"MultiVAE_{ccy}"]  for ccy in CURRENCIES])
rmse_cvae_all   = np.concatenate([rmse_results[f"CVAE_{ccy}"]      for ccy in CURRENCIES])
```


```python
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
```


    
![png](03_vae_training_files/03_vae_training_63_0.png)
    


The hierarchy is correct: CVAE > Multi-VAE > Single-VAE. The CVAE has its peak far to the left (~2 bp), which is excellent. The Single-VAE has a very heavy tail up to 50+ bp — consistent with the individual RMSEs seen in the table (DKK single = 15.75 bp, AUD = 12.97 bp). 


```python
print("\nSummary Statistics:")
print(f"  Single-Currency VAE: Mean={np.mean(rmse_single_all):.1f} bp, Median={np.median(rmse_single_all):.1f} bp")
print(f"  Multi-Currency VAE:  Mean={np.mean(rmse_multi_all):.1f} bp,  Median={np.median(rmse_multi_all):.1f} bp")
print(f"  Multi-Currency CVAE: Mean={np.mean(rmse_cvae_all):.1f} bp,  Median={np.median(rmse_cvae_all):.1f} bp")

```

    
    Summary Statistics:
      Single-Currency VAE: Mean=9.9 bp, Median=7.9 bp
      Multi-Currency VAE:  Mean=5.5 bp,  Median=5.0 bp
      Multi-Currency CVAE: Mean=3.8 bp,  Median=3.1 bp


## 11 - Figure 10: VAE vs Nelson-Siegel Comparison


```python
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
```


    
![png](03_vae_training_files/03_vae_training_67_0.png)
    



```python
print("\nKey Finding:")
print(f"  Nelson-Siegel (3D):      Mean={np.mean(rmse_ns_all):.1f} bp")
print(f"  Multi-Currency VAE (2D): Mean={np.mean(rmse_multi_all):.1f} bp")

```

    
    Key Finding:
      Nelson-Siegel (3D):      Mean=6.8 bp
      Multi-Currency VAE (2D): Mean=5.5 bp


## 12 - Figure 11: RMSE Distribution by Currency



```python
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

```


    
![png](03_vae_training_files/03_vae_training_70_0.png)
    


## 13 - Figure 12: In-Sample vs Out-of-Sample Validation



```python
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
```

    
    ========================================================================
    Training Multi-Currency VAE for Out-of-Sample Validation
    ========================================================================
      Epoch     1/5000 | Loss: 3.295829e-01 | Recon: 3.295829e-01 | KLD: 7.768e-02 | LR: 1.0e-03
      Epoch   500/5000 | Loss: 3.395026e-04 | Recon: 3.379637e-04 | KLD: 1.539e+01 | LR: 1.0e-03
      Epoch  1000/5000 | Loss: 2.915449e-04 | Recon: 2.896842e-04 | KLD: 1.861e+01 | LR: 1.0e-03
      Epoch  1500/5000 | Loss: 2.881789e-04 | Recon: 2.863540e-04 | KLD: 1.825e+01 | LR: 1.0e-03
      Epoch  2000/5000 | Loss: 2.884875e-04 | Recon: 2.867570e-04 | KLD: 1.730e+01 | LR: 1.0e-03
      Epoch  2500/5000 | Loss: 2.868905e-04 | Recon: 2.852613e-04 | KLD: 1.629e+01 | LR: 1.0e-03
      Epoch  3000/5000 | Loss: 2.795828e-04 | Recon: 2.780628e-04 | KLD: 1.520e+01 | LR: 1.0e-03
      Epoch  3500/5000 | Loss: 2.535497e-04 | Recon: 2.521923e-04 | KLD: 1.357e+01 | LR: 1.0e-03
      Epoch  4000/5000 | Loss: 3.663416e-05 | Recon: 3.526452e-05 | KLD: 1.370e+01 | LR: 1.0e-03
      Epoch  4500/5000 | Loss: 3.362490e-05 | Recon: 3.224886e-05 | KLD: 1.376e+01 | LR: 1.0e-03
      Epoch  5000/5000 | Loss: 3.317108e-05 | Recon: 3.179453e-05 | KLD: 1.377e+01 | LR: 1.0e-03
    
    In-sample:     3962 observations, Mean RMSE = 5.45 bp
    Out-of-sample: 1456 observations, Mean RMSE = 6.12 bp



    
![png](03_vae_training_files/03_vae_training_72_1.png)
    


The OOS/IS ratio of 1.12 is well below the 1.5 threshold, indicating no signs of overfitting. The two distributions are almost perfectly overlapping, with only a slightly heavier tail for the OOS case, which is exactly the behavior we want to highlight in the report to validate the model’s ability to generalize. Looking at the training convergence for this OOS model, the loss remains relatively high (~2.8e-4) up to around epoch 3500 before dropping sharply to ~3.5e-5 by epoch 4000. This suggests that a learning rate scheduler was triggered quite late, and the model took a significant amount of time to settle into the right basin. This is not an issue in itself, but it does justify the need for running 5000 epochs in this case.

## 14 - Figure 13: World Map of Latent Space


```python
# Data-driven axis limits (no hard-coded `xlim`/`ylim`) so the plot adapts to whatever
# the encoder actually produces.
default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = {ccy: default_cycle[i % len(default_cycle)] for i, ccy in enumerate(CURRENCIES)}
```


```python
# Gather latent codes per currency
latent_by_ccy = {}
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    latent_by_ccy[ccy] = multi_vae.get_latent(X_multi[offset:offset+n])
    offset += n
```


```python
all_z = np.vstack(list(latent_by_ccy.values()))
```


```python
def axis_lim(vals, margin=0.15):
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * margin
    return lo - pad, hi + pad
```


```python
z1_lim = axis_lim(all_z[:, 0])
z2_lim = axis_lim(all_z[:, 1])
```


```python
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
```


    
![png](03_vae_training_files/03_vae_training_80_0.png)
    


This is the most conceptually rich figure. JPY appears completely isolated in the bottom-right region (z1≈0.39, z2≈1.34), far from the other currencies, which is consistent with all prior observations: it operates under a fundamentally different rate regime (near-zero rates for years followed by BoJ normalization), and the VAE has effectively learned to separate it. The other six currencies overlap within the region z1∈[-0.2, 0.5] and z2∈[1.6, 2.1], reflecting the similarity in their curve shapes, as they all belong to positive-rate regimes with synchronized post-COVID tightening and easing cycles. The ellipses (panel b) further highlight this structure: EUR and DKK are close, consistent with their economic linkage; GBP and AUD display wider spreads, indicating greater shape volatility; and CAD and USD partially overlap, reflecting Fed–BoC synchronization. All of these patterns are economically interpretable and provide a strong argument for the report. Overall, the key takeaway is that the VAE spontaneously organizes the latent space according to economic regimes—isolating JPY due to its unique monetary policy, while clustering Western currencies into a dense group that reflects their post-COVID synchronization.



```python
print("\nLatent code summary (mean ± std):")
print(f"  {'':6} {'z1 mean':>9} {'z1 std':>9} | {'z2 mean':>9} {'z2 std':>9}")
for ccy, z in latent_by_ccy.items():
    print(f"  {ccy:6} {z[:,0].mean():>9.4f} {z[:,0].std():>9.4f} | {z[:,1].mean():>9.4f} {z[:,1].std():>9.4f}")

```

    
    Latent code summary (mean ± std):
               z1 mean    z1 std |   z2 mean    z2 std
      AUD       0.2473    0.0647 |    1.8623    0.0406
      CAD       0.1624    0.1259 |    1.7637    0.0814
      DKK       0.1580    0.1098 |    1.6904    0.0609
      EUR       0.1819    0.1051 |    1.6382    0.0594
      GBP       0.1372    0.1332 |    1.8841    0.0490
      JPY       0.3922    0.0328 |    1.3431    0.0695
      USD       0.1194    0.1101 |    1.8415    0.0477


All currencies have z2 values between 1.34 and 1.88, which are clearly far from zero. In a standard VAE, the prior is N(0,1), so latent variables are typically expected to be centered around zero; here, however, z2 is consistently in the 1.6–1.9 range. This indicates that the KLD term (β=1e-7) is too weak to pull the latent codes toward the prior, meaning the VAE behaves almost like a standard autoencoder. This is in fact intentional in Sokol’s approach (using a very small β to prioritize reconstruction), but it also implies that the latent space is not centered around zero. This should be explicitly mentioned as a deliberate modeling choice.


## 15 - Figure 14: Historical vs Reconstructed Curves (Spaghetti Plots)



```python
tenors = np.array(TARGET_TENORS, dtype=float)
representative_ccys = CURRENCIES 
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
```


    
![png](03_vae_training_files/03_vae_training_85_0.png)
    


The overall shapes are very well preserved. The main difference between the historical data and the reconstruction lies in the reduction of extreme variance. The VAE compresses outliers toward more average shapes. This is a fundamental property of VAE regularization and should be highlighted in the report.

## 16 - Figure 15: Generated Curves from Latent Space Perimeter


```python
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
```


    
![png](03_vae_training_files/03_vae_training_88_0.png)
    


AUD shows a compact 2σ region (z1 ∈ [0.15, 0.45]), with decoded curves spanning levels from 3.0% to 4.75%, effectively covering the full historical AUD range. The z2 color gradient indicates that higher z2 (blue) corresponds to higher rate levels, meaning z2 clearly encodes the level dimension for AUD.

For CAD, the ellipse is much more horizontally stretched, indicating strong variation in z1. The decoded curves include both inverted and normal shapes, showing that z1 captures the curve shape (inverted vs normal) for CAD. This is a key result: z1 behaves as a slope/shape factor, while z2 captures the level—an interpretation that should be made explicit.

EUR also exhibits an elongated ellipse, with curves ranging from 1.5% to 3.5%. The color mapping again confirms that z2 drives the level, while the shapes themselves remain relatively similar (mostly moderately inverted or flat), indicating less shape diversity compared to CAD.

GBP displays a very wide horizontal spread, with decoded curves ranging from 3.2% to 5.5% and showing highly diverse shapes—some strongly inverted, others more normal. This suggests that GBP experienced the greatest diversity of regimes over the period, consistent with the large variation observed in z1.

JPY is the most informative case: its ellipse is completely separated from the others (bottom-right region), and the decoded curves range from 0% to 2.5%, making it the only country with near-zero rates. The z2 color range (1.25 to 1.45) is entirely distinct from other currencies (all above 1.6), indicating that the VAE has effectively carved out a dedicated latent region for the ultra-low-rate regime of the BoJ. This is a particularly strong argument to highlight in the slides.

Finally, USD shows a broad ellipse with curves spanning 3% to 5% and a variety of shapes, including inverted ones. As with the other currencies, z2 encodes the level (with higher values corresponding to higher rates), reinforcing the consistency of this latent interpretation across datasets.


## 17 - Forward Rate AEMM (Section 3.1.3)

Chapter 3: Q-Measure AEMM Models
1. **Forward Rate AEMM** (Section 3.1.3)
2. **Multi-Factor Short Rate AEMM** (Section 3.2.2)



```python
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
```


```python
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
```

    
    ========================================================================
    Testing Forward Rate AEMM
    ========================================================================
    
    Initial curve (AUD): [3.618  3.5251 3.5076 3.8066 3.768  3.856  3.538 ]
    
    Simulated 100 paths over 12 time steps
    Latent paths shape: (100, 13, 2)
    Curve paths shape:  (100, 13, 7)



    
![png](03_vae_training_files/03_vae_training_93_1.png)
    


## 18 - Multi-Factor Short Rate AEMM (Section 3.2.2)


```python
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
```


```python
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
```

    
    ========================================================================
    Testing Short Rate AEMM
    ========================================================================
    
    Simulated 500 paths over 60 months



    
![png](03_vae_training_files/03_vae_training_96_1.png)
    


## 19 - Autoregressive AEMM (Section 4.1.2)


Chapter 4: P-Measure AEMM Models
1. **Autoregressive AEMM** (Section 4.1.2) — replacement for Dynamic Nelson-Siegel
2. **VAR(1)** on latent factors


```python
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
```


```python
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
```

    
    ========================================================================
    Fitting Autoregressive AEMM
    ========================================================================
    VAR(1) fitted on 774 observations
    Intercept c: [0.04304722 0.00942742]
    AR matrix A eigenvalues: [0.98955673+0.00529332j 0.98955673-0.00529332j]
    
    Generated 1000 forecast paths for 12 periods



    
![png](03_vae_training_files/03_vae_training_100_1.png)
    


Eigenvalue modulus = 0.990 < 1 confirms VAR(1) stationarityt. The latent factors exhibit strong but mean-reverting persistence, consistent with interest rate dynamics.

## 19b - Statistical Comparison & Final Findings

This section consolidates the comparison NS vs Multi-VAE vs CVAE with:
- Per-currency Wilcoxon signed-rank tests (paired RMSE)
- Global pooled Wilcoxon test
- Summary table (mean / median / p95) by method × currency
- Final replication verdict


```python
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
```

    RMSE comparison table (bp):
    Currency          Method   N  Mean(bp)  Median(bp)   P95(bp)
         AUD       CVAE (2D) 774  4.001137    3.386528  8.669450
         AUD  Multi VAE (2D) 774  5.766629    5.004000 11.464496
         AUD         NS (3D) 774 10.000478    9.465532 16.229522
         AUD Single VAE (2D) 774 12.970739   12.580370 23.715436
         CAD       CVAE (2D) 774  4.534052    4.168195  9.134596
         CAD  Multi VAE (2D) 774  6.099432    5.435401 13.347836
         CAD         NS (3D) 774  7.427125    6.995878 12.046328
         CAD Single VAE (2D) 774 11.847674   10.126320 24.157453
         DKK       CVAE (2D) 774  4.876113    4.134624 11.453491
         DKK  Multi VAE (2D) 774  6.284379    5.741586 13.078813
         DKK         NS (3D) 774  8.307055    8.016556 15.525043
         DKK Single VAE (2D) 774 15.748870   14.408533 31.271571
         EUR       CVAE (2D) 774  2.753947    2.429981  5.512874
         EUR  Multi VAE (2D) 774  3.377508    3.063704  6.217548
         EUR         NS (3D) 774  6.347344    6.498285  8.798537
         EUR Single VAE (2D) 774 13.201626   11.852324 28.929434
         GBP       CVAE (2D) 774  2.488935    2.032022  5.827597
         GBP  Multi VAE (2D) 774  6.780932    6.638595  9.897637
         GBP         NS (3D) 774  2.527322    2.480993  3.194069
         GBP Single VAE (2D) 774  2.482599    1.968184  5.643342
         JPY       CVAE (2D) 774  2.998201    2.710254  5.657523
         JPY  Multi VAE (2D) 774  4.627429    4.526863  8.172089
         JPY         NS (3D) 774  6.492899    6.096812 11.664342
         JPY Single VAE (2D) 774  4.990792    4.706696  9.876825
         USD       CVAE (2D) 774  4.644120    3.773752 10.129559
         USD  Multi VAE (2D) 774  5.515566    4.585840 11.825042
         USD         NS (3D) 774  6.279040    5.699949 10.798787
         USD Single VAE (2D) 774  8.114661    6.978661 15.703536


GBP is the sole exception where NS outperforms Multi-VAE. This reflects a fundamental trade-off of the shared latent space: GBP's simple, regular curve shapes are well captured by the 3-parameter NS form, while the multi-currency VAE sacrifices GBP accuracy to better represent more complex currencies.


```python
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
```

    
    ========================================================================
    Wilcoxon Signed-Rank Test: Multi-VAE (2D) vs Nelson-Siegel (3D)
    H0: median paired RMSE difference = 0
    ========================================================================
      AUD: meanΔ=  +4.23 bp   p=1.02e-127   [Multi-VAE better]
      CAD: meanΔ=  +1.33 bp   p=1.05e-42   [Multi-VAE better]
      DKK: meanΔ=  +2.02 bp   p=2.67e-60   [Multi-VAE better]
      EUR: meanΔ=  +2.97 bp   p=1.25e-122   [Multi-VAE better]
      GBP: meanΔ=  -4.25 bp   p=2.42e-128   [NS better]
      JPY: meanΔ=  +1.87 bp   p=3.01e-42   [Multi-VAE better]
      USD: meanΔ=  +0.76 bp   p=2.05e-25   [Multi-VAE better]
    
    ========================================================================
    Wilcoxon Signed-Rank Test: CVAE (2D) vs Nelson-Siegel (3D)
    ========================================================================
      AUD: meanΔ=  +6.00 bp   p=2.96e-128   [CVAE better]
      CAD: meanΔ=  +2.89 bp   p=9.87e-112   [CVAE better]
      DKK: meanΔ=  +3.43 bp   p=4.83e-108   [CVAE better]
      EUR: meanΔ=  +3.59 bp   p=7.59e-126   [CVAE better]
      GBP: meanΔ=  +0.04 bp   p=1.07e-05   [CVAE better]
      JPY: meanΔ=  +3.49 bp   p=8.40e-109   [CVAE better]
      USD: meanΔ=  +1.63 bp   p=7.59e-83   [CVAE better]



```python
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
```

    
    ========================================================================
    Pooled Wilcoxon (all currencies)
    ========================================================================
      NS vs Multi-VAE: meanΔ=+1.28 bp   p=2.64e-154   [Multi-VAE better]
      NS vs CVAE     : meanΔ=+3.01 bp   p=0.00e+00   [CVAE better]



```python
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
```

    
    ========================================================================
      REPLICATION SUMMARY — Sokol (2022) AEMM
    ========================================================================
    
      Sample: 7 currencies × 7 tenors, 5418 total observations
      Train cutoff: 2025-03-31
    
      Mean RMSE ranges (across currencies):
        Nelson-Siegel (3D)  : 2.53–10.00 bp
        Single VAE    (2D)  : 2.48–15.75 bp
        Multi VAE     (2D)  : 3.38–6.78 bp
        CVAE          (2D)  : 2.49–4.88 bp
    
      Multi-VAE beats NS: 6/7 currencies
      CVAE      beats NS: 7/7 currencies
    
      Out-of-sample sanity:
        In-sample  mean RMSE : 5.45 bp
        Out-sample mean RMSE : 6.12 bp
        Ratio OOS/IS         : 1.12×  (no overfitting)
    
      Key findings:
        1. Multi-VAE (2D) achieves comparable accuracy to NS (3D) with one fewer latent dimension.
        2. The shared latent space lets one model handle all 7 currencies simultaneously.
        3. Latent dimensions z1/z2 separate currencies by rate regime (see Figure 13).
    ========================================================================





## 20 - Save All Results


```python
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
```
