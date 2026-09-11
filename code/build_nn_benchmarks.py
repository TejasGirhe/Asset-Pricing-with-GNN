"""
Build the NN2 / NN3 (feed-forward MLP) non-graph benchmarks for the Table 5
spanning regression.

Paper spec (paper_text.txt lines ~839-865, Eq. 14):
  "We evaluate NN architectures with L in {2, 3} hidden layers, where each hidden
   layer contains 40 neurons. ... estimated by minimizing the mean squared error
   loss over the training set D_train using stochastic gradient descent with Adam
   optimization, with learning rate selected via grid search and early stopping
   based on validation set performance."

Hyperparameter choices (documented judgment calls):
  - hidden_dim = 40 neurons per hidden layer  -> stated in the paper.
  - num_hidden_layers in {2, 3}                -> NN2 / NN3, stated in the paper.
  - embedding_dim = 40                         -> matches the GNN's output/embedding
        dimension (paper's selected config, line 805-806) and the value used in
        GNN_code_additional.ipynb cell 5/6. The paper's Eq. (14) has the last
        hidden layer feed directly into a scalar; our MLPModel inserts a 40-wide
        "embedding" head before the scalar output purely to mirror the GNN code
        path. With hidden_dim already 40 this is an identity-width extra ReLU
        layer and does not change the model class materially.
  - dropout = 0.0  -> the paper does NOT specify a dropout rate for the NN
        benchmark (the 0.2/0.3/0.5/0.8 dropout grid in footnote 14 is for the GNN
        only). The paper regularizes the NN via "early stopping based on
        validation set performance" plus Adam weight decay. We follow
        GNN_code_additional.ipynb cells 5/6 (dropout=0) and rely on
        weight_decay=1e-4 + a fixed 80-epoch budget matching the GNN. Documented
        here so the choice is explicit.
  - epochs = 80, batch_size = 32, lr = 1e-3, weight_decay = 1e-4  -> from
        GNN_code_additional.ipynb cells 5/6, consistent with the GNN's selected
        80-epoch / lr-1e-3 config. We do not implement a separate validation
        hold-out for early stopping; instead we use the paper's 80-epoch budget
        directly (the paper reports convergence -- "training loss remains
        relatively constant" -- by the final epochs, line 826).

Target / split (must match the NC Ridge/LASSO benchmark in GNN model.ipynb so
NN2/NN3 are directly comparable in the same spanning regression):
  - target      = mthret - rf   (excess return; GNN model.ipynb cell 41:
                  df['true_return'] = df['mthret'] - df['rf'])
  - feature_cols = the 64 characteristic columns AT..Total_vol
  - train/test split at yyyymm <= 201612 / > 201612 (paper's split).

Everything is seeded (torch / numpy / random = 42) for reproducibility.

Outputs (wide: index = yyyymm, columns = gvkey as plain int, values = predicted
excess return; both train and test period), matching
data/firm_level_reg_pca_predictions_ALL_firms.csv exactly:
  - data/firm_level_NN2_predictions_ALL_firms.csv
  - data/firm_level_NN3_predictions_ALL_firms.csv
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(NOTEBOOK_DIR)
DATA = "../data"

SEED = 42
SPLIT_YYYYMM = 201612
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.0
EMBEDDING_DIM = 40
HIDDEN_DIM = 40


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


class PanelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLPModel(nn.Module):
    """Mirrors GNN_code_additional.ipynb cell 3's MLPModel: BN on raw inputs,
    `num_hidden_layers` x (Linear->ReLU->Dropout), a `embedding_dim`-wide ReLU
    embedding head, then a scalar output head."""

    def __init__(self, input_dim, embedding_dim=40, num_hidden_layers=2,
                 hidden_dim=40, dropout=0.0, use_batchnorm=True):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(input_dim) if use_batchnorm else None
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.to_embedding = nn.Linear(in_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        if self.bn_in is not None:
            x = self.bn_in(x)
        x = self.hidden(x)
        emb = torch.relu(self.to_embedding(x))
        return self.out(emb)


def load_panel():
    df = pd.read_csv(f"{DATA}/features.csv")
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    df = df.drop_duplicates()
    # 64 characteristic columns: everything after the 7 metadata cols
    # (yyyymm, permno, gvkey, exchcd, mthret, rf, sic2)
    meta = ["yyyymm", "permno", "gvkey", "exchcd", "mthret", "rf", "sic2"]
    feature_cols = [c for c in df.columns if c not in meta]
    assert len(feature_cols) == 64, f"expected 64 characteristics, got {len(feature_cols)}"
    df["target"] = df["mthret"] - df["rf"]
    df = df.dropna(subset=["target", "yyyymm", "gvkey"] + feature_cols)
    df["yyyymm"] = df["yyyymm"].astype(int)
    df["gvkey"] = df["gvkey"].astype(int)
    return df, feature_cols


def train_one(df, feature_cols, num_hidden_layers):
    seed_everything(SEED)

    train_df = df[df["yyyymm"] <= SPLIT_YYYYMM]
    test_df = df[df["yyyymm"] > SPLIT_YYYYMM]

    X_tr = train_df[feature_cols].to_numpy(np.float32)
    y_tr = train_df["target"].to_numpy(np.float32)
    X_te = test_df[feature_cols].to_numpy(np.float32)
    y_te = test_df["target"].to_numpy(np.float32)

    g = torch.Generator()
    g.manual_seed(SEED)
    # drop_last=True: BatchNorm1d errors on a trailing batch of size 1
    # (699,201 train rows % 32 == 1); dropping it loses one row per epoch, harmless.
    train_loader = DataLoader(PanelDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                              shuffle=True, generator=g, drop_last=True)

    model = MLPModel(input_dim=len(feature_cols), embedding_dim=EMBEDDING_DIM,
                     num_hidden_layers=num_hidden_layers, hidden_dim=HIDDEN_DIM,
                     dropout=DROPOUT, use_batchnorm=True)
    opt = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        tot = 0.0
        for Xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(Xb), yb.unsqueeze(1))
            loss.backward()
            opt.step()
            tot += loss.item()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                te_loss = crit(model(torch.tensor(X_te)), torch.tensor(y_te).unsqueeze(1)).item()
            print(f"  NN{num_hidden_layers} epoch {epoch+1:>3}/{EPOCHS}  "
                  f"train_mse={tot/len(train_loader):.6e}  test_mse={te_loss:.6e}  "
                  f"({time.time()-t0:.0f}s)")
    elapsed = time.time() - t0

    # inference on the full panel (train + test period)
    model.eval()
    all_df = df[["yyyymm", "gvkey"]].copy()
    X_all = df[feature_cols].to_numpy(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_all), 8192):
            preds.append(model(torch.tensor(X_all[i:i + 8192])).squeeze(1).numpy())
    all_df["pred"] = np.concatenate(preds)
    return all_df, elapsed


def to_wide_like_pca(long_df):
    """Match firm_level_reg_pca_predictions_ALL_firms.csv: rows = all 264 yyyymm
    in features.csv order, columns = every gvkey (plain int header), one prediction
    per firm-month (mean if a firm has dup rows in a month)."""
    ref = pd.read_csv(f"{DATA}/firm_level_reg_pca_predictions_ALL_firms.csv", nrows=0)
    ref_months = pd.read_csv(f"{DATA}/firm_level_reg_pca_predictions_ALL_firms.csv",
                             usecols=["yyyymm"])["yyyymm"].astype(int).tolist()
    ref_gvkeys = [int(c) for c in ref.columns if c != "yyyymm"]

    wide = (long_df.groupby(["yyyymm", "gvkey"])["pred"].mean()
            .unstack("gvkey"))
    wide = wide.reindex(index=ref_months, columns=ref_gvkeys)
    wide.index.name = "yyyymm"
    return wide


def main():
    df, feature_cols = load_panel()
    print(f"panel: {len(df):,} rows, {df['gvkey'].nunique()} firms, "
          f"{df['yyyymm'].nunique()} months; target = mthret - rf; "
          f"{len(feature_cols)} features")
    print(f"train (<= {SPLIT_YYYYMM}): {(df['yyyymm'] <= SPLIT_YYYYMM).sum():,} rows | "
          f"test (> {SPLIT_YYYYMM}): {(df['yyyymm'] > SPLIT_YYYYMM).sum():,} rows")

    for depth, name in [(2, "NN2"), (3, "NN3")]:
        print(f"\n=== training {name} ({depth} hidden layers x {HIDDEN_DIM} neurons) ===")
        long_df, elapsed = train_one(df, feature_cols, depth)
        wide = to_wide_like_pca(long_df)
        out = f"{DATA}/firm_level_{name}_predictions_ALL_firms.csv"
        wide.to_csv(out, encoding="utf-8")
        print(f"  {name} trained in {elapsed:.0f}s ({elapsed/60:.1f} min); "
              f"wrote {out}  shape={wide.shape[0]+0, wide.shape[1]+1} "
              f"(non-null cells: {int(wide.notna().sum().sum()):,})")


if __name__ == "__main__":
    main()
