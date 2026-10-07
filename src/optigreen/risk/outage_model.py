"""
Refinery-district outage risk: P(outage in month t | information up to origin o-1).

Label  : optigreen.data.realcase.outage_labels (district utilisation falls at
         least 10 points more than the national change, vs its own trailing
         12-month median). Real events it captures include Katrina/Rita (2005),
         Ike (2008), Harvey (2017), the Philadelphia Energy Solutions fire
         (2019), Laura (2020), Winter Storm Uri (Feb 2021) and Ida (2021).
Horizon: h = 1, 2, 3 months ahead (same planning window as the optimiser).

Models
------
* climatology : expanding frequency per district x calendar month (Laplace
                smoothed), using only outages observed before the origin
* logistic    : L2 logistic regression on the tabular features
* xgboost     : gradient-boosted trees on the same features
* gat         : graph attention network over the 12-district graph; each
                snapshot (origin, h) is one graph, node features are the same
                tabular features, edges join districts in the same PADD or
                within 900 km (shared weather and pipeline systems)
* gat_self    : identical GAT with self-loops only (ablation: no message passing)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

try:  # PyTorch is only needed to train or run the GAT; the dashboard runs without it
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _Module = nn.Module
except ImportError:  # pragma: no cover - exercised by the lean deployment image
    torch = nn = F = None
    _Module = object
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from optigreen.data.network import COASTAL_DISTRICTS, DISTRICT_INFO, haversine_km

DISTRICTS = list(DISTRICT_INFO.keys())
GULF = {"3B", "3C"}
FEATURES = [
    "h", "sin_t", "cos_t", "hurricane_season_gulf", "hurricane_season_coastal", "winter_south",
    "log_capacity", "util_last", "idio_last", "idio_min3", "idio_min12", "util_std12",
    "months_since_outage", "outages_24", "outages_60", "clim_rate", "dist_rate",
    "us_dev_last", "cap_change_12",
] + [f"d_{d}" for d in DISTRICTS]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_rows(labels: pd.DataFrame, origins: Sequence[pd.Timestamp],
               horizons: Sequence[int] = (1, 2, 3)) -> pd.DataFrame:
    lab = labels.pivot_table(index="month", columns="district", values="outage")
    util = labels.pivot_table(index="month", columns="district", values="utilization_pct")
    idio = labels.pivot_table(index="month", columns="district", values="idio_dev")
    cap = labels.pivot_table(index="month", columns="district", values="capacity_kbcd")
    sev = labels.pivot_table(index="month", columns="district", values="severity")
    usdev = labels.drop_duplicates("month").set_index("month")["us_dev"].sort_index()
    months = lab.index
    pos = {m: i for i, m in enumerate(months)}
    rows = []
    for o in origins:
        if o not in pos or pos[o] < 36:
            continue
        io = pos[o]
        past_lab = lab.iloc[:io]
        for d in DISTRICTS:
            pl = past_lab[d].dropna()
            if len(pl) < 24:
                continue
            ev = pl[pl == 1]
            msince = (io - pos[ev.index[-1]]) if len(ev) else 240
            cl = past_lab[[d]].assign(m=past_lab.index.month)
            dist_rate = (pl.sum() + 0.5) / (len(pl) + 1.0)
            for h in horizons:
                it = io + h - 1
                t = o + pd.DateOffset(months=h - 1)
                same_m = cl[cl["m"] == t.month][d].dropna()
                clim = (same_m.sum() + dist_rate * 2) / (len(same_m) + 2)
                y = lab[d].iloc[it] if it < len(months) else np.nan
                s = sev[d].iloc[it] if it < len(months) else np.nan
                u_hist = util[d].iloc[:io]
                c_now = cap[d].iloc[io - 1]
                c_prev = cap[d].iloc[max(0, io - 13)]
                r = {
                    "origin": o, "target_month": t, "district": d, "h": h,
                    "sin_t": math.sin(2 * math.pi * t.month / 12), "cos_t": math.cos(2 * math.pi * t.month / 12),
                    "hurricane_season_gulf": float(d in GULF and t.month in (8, 9, 10)),
                    "hurricane_season_coastal": float(d in COASTAL_DISTRICTS and t.month in (6, 7, 8, 9, 10, 11)),
                    "winter_south": float(d.startswith("3") or d == "2C") * float(t.month in (12, 1, 2)),
                    "log_capacity": math.log1p(c_now) if pd.notna(c_now) else 0.0,
                    "util_last": u_hist.iloc[-1], "idio_last": idio[d].iloc[io - 1],
                    "idio_min3": idio[d].iloc[io - 3:io].min(), "idio_min12": idio[d].iloc[io - 12:io].min(),
                    "util_std12": u_hist.iloc[-12:].std(),
                    "months_since_outage": min(msince, 240), "outages_24": pl.iloc[-24:].sum(),
                    "outages_60": pl.iloc[-60:].sum(), "clim_rate": clim, "dist_rate": dist_rate,
                    "us_dev_last": usdev.iloc[io - 1] if io - 1 < len(usdev) else np.nan,
                    "cap_change_12": (c_now / c_prev - 1) if (pd.notna(c_now) and pd.notna(c_prev) and c_prev > 0) else 0.0,
                    "y": y, "severity": s,
                }
                for dd in DISTRICTS:
                    r[f"d_{dd}"] = float(dd == d)
                rows.append(r)
    df = pd.DataFrame(rows)
    num = [c for c in FEATURES if c in df]
    df[num] = df[num].astype(float)
    return df


def district_graph(max_km: float = 900.0, self_only: bool = False) -> np.ndarray:
    n = len(DISTRICTS)
    A = np.eye(n, dtype=np.float32)
    if self_only:
        return A
    for i, a in enumerate(DISTRICTS):
        for j, b in enumerate(DISTRICTS):
            if i == j:
                continue
            pa, la, oa = DISTRICT_INFO[a][0], DISTRICT_INFO[a][1], DISTRICT_INFO[a][2]
            pb, lb, ob = DISTRICT_INFO[b][0], DISTRICT_INFO[b][1], DISTRICT_INFO[b][2]
            if pa == pb or haversine_km(la, oa, lb, ob) <= max_km:
                A[i, j] = 1.0
    return A


# --------------------------------------------------------------------------- #
# Graph attention network (dense, pure PyTorch: 12 nodes per snapshot)
# --------------------------------------------------------------------------- #
def _require_torch() -> None:
    if torch is None:
        raise ImportError("The graph attention network needs PyTorch: "
                          "pip install torch --index-url https://download.pytorch.org/whl/cpu")


class DenseGATLayer(_Module):
    def __init__(self, f_in: int, f_out: int, heads: int = 2, dropout: float = 0.1):
        _require_torch()
        super().__init__()
        self.heads, self.f_out = heads, f_out
        self.W = nn.Linear(f_in, heads * f_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, f_out))
        self.a_dst = nn.Parameter(torch.empty(heads, f_out))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.f_out)                    # B N H F
        e_src = (h * self.a_src).sum(-1)                                     # B N H
        e_dst = (h * self.a_dst).sum(-1)
        e = F.leaky_relu(e_dst.unsqueeze(2) + e_src.unsqueeze(1), 0.2)       # B N(dst) N(src) H
        mask = (adj > 0).view(1, N, N, 1)
        e = e.masked_fill(~mask, float("-inf"))
        att = torch.softmax(e, dim=2)
        self.last_attention = att.detach()
        att = self.drop(att)
        out = torch.einsum("bijh,bjhf->bihf", att, h)                        # B N H F
        return out.reshape(B, N, self.heads * self.f_out)


class DistrictGAT(_Module):
    def __init__(self, f_in: int, hidden: int = 16, heads: int = 2, dropout: float = 0.1):
        _require_torch()
        super().__init__()
        self.inp = nn.Linear(f_in, hidden)
        self.gat1 = DenseGATLayer(hidden, hidden, heads, dropout)
        self.gat2 = DenseGATLayer(hidden * heads, hidden, 1, dropout)
        self.out = nn.Sequential(nn.Linear(hidden + hidden, 16), nn.ELU(), nn.Linear(16, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h0 = F.elu(self.inp(x))
        h = F.elu(self.gat1(self.drop(h0), adj))
        h = F.elu(self.gat2(self.drop(h), adj))
        return self.out(torch.cat([h, h0], dim=-1)).squeeze(-1)  # B N logits


def _to_snapshots(rows: pd.DataFrame, cols: List[str], scaler: StandardScaler):
    keys = rows[["origin", "h"]].drop_duplicates().sort_values(["origin", "h"]).to_numpy()
    n = len(DISTRICTS)
    X = np.zeros((len(keys), n, len(cols)), dtype=np.float32)
    Y = np.full((len(keys), n), np.nan, dtype=np.float32)
    M = np.zeros((len(keys), n), dtype=bool)
    idx = {(o, h): i for i, (o, h) in enumerate(map(tuple, keys))}
    dpos = {d: i for i, d in enumerate(DISTRICTS)}
    Z = scaler.transform(rows[cols].fillna(0.0).to_numpy())
    for r, z in zip(rows[["origin", "h", "district", "y"]].itertuples(index=False), Z):
        i, j = idx[(r.origin, r.h)], dpos[r.district]
        X[i, j] = z
        if not np.isnan(r.y):
            Y[i, j] = r.y
            M[i, j] = True
    return keys, torch.tensor(X), torch.tensor(np.nan_to_num(Y)), torch.tensor(M)


@dataclass
class GATRisk:
    self_only: bool = False
    hidden: int = 16
    heads: int = 2
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 30
    seed: int = 42

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "GATRisk":
        _require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.cols = FEATURES
        self.scaler = StandardScaler().fit(train[self.cols].fillna(0.0).to_numpy())
        self.adj = torch.tensor(district_graph(self_only=self.self_only))
        _, Xtr, Ytr, Mtr = _to_snapshots(train, self.cols, self.scaler)
        _, Xva, Yva, Mva = _to_snapshots(val, self.cols, self.scaler)
        self.model = DistrictGAT(len(self.cols), self.hidden, self.heads)
        pos = float(Ytr[Mtr].mean())
        pw = torch.tensor((1 - pos) / max(pos, 1e-3))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad = float("inf"), None, 0
        self.history = []
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(Xtr.shape[0])
            for b in range(0, len(perm), 64):
                ib = perm[b:b + 64]
                logit = self.model(Xtr[ib], self.adj)
                loss = F.binary_cross_entropy_with_logits(logit[Mtr[ib]], Ytr[ib][Mtr[ib]], pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                lv = F.binary_cross_entropy_with_logits(self.model(Xva, self.adj)[Mva], Yva[Mva], pos_weight=pw).item()
            self.history.append(lv)
            if lv < best - 1e-4:
                best, bad = lv, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch = int(np.argmin(self.history)) + 1
        # Platt recalibration on validation (pos_weight inflates raw probabilities)
        with torch.no_grad():
            lv = self.model(Xva, self.adj)[Mva].numpy()
        self.platt = LogisticRegression(C=1e6).fit(lv.reshape(-1, 1), Yva[Mva].numpy())
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        keys, X, _, _ = _to_snapshots(rows, self.cols, self.scaler)
        self.model.eval()
        with torch.no_grad():
            logit = self.model(X, self.adj).numpy()
        idx = {(o, h): i for i, (o, h) in enumerate(map(tuple, keys))}
        dpos = {d: i for i, d in enumerate(DISTRICTS)}
        raw = np.array([logit[idx[(o, h)], dpos[d]] for o, h, d in
                        rows[["origin", "h", "district"]].itertuples(index=False)])
        return self.platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    def attention(self, rows: pd.DataFrame) -> np.ndarray:
        """Mean attention matrix (dst x src) of the first GAT layer."""
        _, X, _, _ = _to_snapshots(rows, self.cols, self.scaler)
        with torch.no_grad():
            self.model(X, self.adj)
        return self.model.gat1.last_attention.mean(dim=(0, 3)).numpy()


# --------------------------------------------------------------------------- #
# Tabular models
# --------------------------------------------------------------------------- #
class TabularRisk:
    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "TabularRisk":
        Xtr, ytr = train[FEATURES].fillna(0.0).to_numpy(), train["y"].to_numpy()
        Xva, yva = val[FEATURES].fillna(0.0).to_numpy(), val["y"].to_numpy()
        if self.kind == "logistic":
            self.scaler = StandardScaler().fit(Xtr)
            self.model = LogisticRegression(C=0.3, max_iter=2000).fit(self.scaler.transform(Xtr), ytr)
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=5.0, eval_metric="logloss", early_stopping_rounds=50,
                random_state=42, n_jobs=2)
            self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        X = rows[FEATURES].fillna(0.0).to_numpy()
        if self.kind == "logistic":
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]


def climatology(rows: pd.DataFrame) -> np.ndarray:
    return rows["clim_rate"].to_numpy()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def risk_metrics(y: np.ndarray, p: np.ndarray, base: Optional[np.ndarray] = None) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "positives": int(y.sum()), "ROC_AUC": float(roc_auc_score(y, p)),
           "AP": float(average_precision_score(y, p)), "Brier": float(brier_score_loss(y, p)),
           "LogLoss": float(log_loss(y, p))}
    if base is not None:
        b = brier_score_loss(y, np.clip(base, 1e-6, 1 - 1e-6))
        out["BrierSkill_vs_clim"] = float(1 - out["Brier"] / b)
    return out


def block_bootstrap_ci(df: pd.DataFrame, cols: List[str], metric: str = "AP", n_boot: int = 500,
                       seed: int = 0) -> Dict[str, Tuple[float, float, float]]:
    """Moving-block bootstrap over target months (block = 6 months) for metric CIs
    and for the difference of each model vs the first column."""
    rng = np.random.default_rng(seed)
    months = np.sort(df["target_month"].unique())
    blocks = [months[i:i + 6] for i in range(0, len(months) - 5)]
    by_m = {m: g for m, g in df.groupby("target_month")}
    fn = {"AP": average_precision_score, "ROC_AUC": roc_auc_score}[metric]
    res = {c: [] for c in cols}
    diffs = {c: [] for c in cols[1:]}
    nb = int(np.ceil(len(months) / 6))
    for _ in range(n_boot):
        pick = [blocks[i] for i in rng.integers(0, len(blocks), nb)]
        sample = pd.concat([by_m[m] for b in pick for m in b])
        if sample["y"].nunique() < 2:
            continue
        vals = {c: fn(sample["y"], sample[c]) for c in cols}
        for c in cols:
            res[c].append(vals[c])
        for c in cols[1:]:
            diffs[c].append(vals[c] - vals[cols[0]])
    out = {c: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for c, v in res.items()}
    for c, v in diffs.items():
        out[f"{c}_minus_{cols[0]}"] = (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out
