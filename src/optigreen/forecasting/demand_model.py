"""
Monthly probabilistic demand forecaster (P10 / P50 / P90) for the real-data case.

Design
------
* Direct multi-horizon: one global model per quantile with the horizon h as a
  feature. A forecast made at origin o for month t = o + h - 1 only uses data up
  to month o - 1. Nothing is fed back recursively, so there is no look-ahead.
* Scale-free target: every series is divided by its trailing 12-month mean
  (level L) so states with very different volumes share one model; training
  rows are weighted by L so large markets dominate, which matches WAPE.
* Conformalised quantile regression (Romano et al., 2019): the P10-P90 band is
  widened or narrowed on a held-out calibration window so its empirical
  coverage matches the nominal 80 %.

Baselines: seasonal naive (same month last year) and seasonal naive with
growth (same month last year x last-3-months / same-3-months-last-year).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

QUANTILES = (0.1, 0.5, 0.9)
HORIZONS = (1, 2, 3)
PRODUCT_CODES = {"GAS": 0, "DIS": 1, "JET": 2, "RES": 3, "LPG": 4}
AREA_CODES = {"1A": 0, "1B": 1, "1C": 2, "P2": 3, "P3": 4, "P4": 5, "P5": 6}
FEATURES = [
    "h", "month_t", "sin_t", "cos_t", "product", "area", "is_pool", "log_level",
    "r_lag1", "r_lag2", "r_lag3", "r_seas12", "r_seas24", "r_seas12_prev", "r_seas12_next",
    "r_mean3", "r_mean6", "r_std12", "yoy3", "trend12",
]
MIN_LEVEL_KT = 0.5  # series below this trailing mean are forecast as their recent mean


def _wide(panel: pd.DataFrame) -> pd.DataFrame:
    """panel(long: month, region, product, kt) -> wide (month x series)."""
    w = panel.pivot_table(index="month", columns=["region", "product"], values="kt", aggfunc="sum")
    return w.sort_index()


def build_rows(panel: pd.DataFrame, origins: Sequence[pd.Timestamp],
               horizons: Sequence[int] = HORIZONS, with_target: bool = True) -> pd.DataFrame:
    """Feature rows for every series x origin x horizon."""
    w = _wide(panel)
    meta = (panel.drop_duplicates(["region", "product"])
            .set_index(["region", "product"])[["area", "kind"]])
    idx = {m: i for i, m in enumerate(w.index)}
    Y = w.to_numpy(dtype=float)  # (n_months, n_series)
    cols = list(w.columns)
    area = np.array([AREA_CODES[meta.loc[c, "area"]] for c in cols])
    prod = np.array([PRODUCT_CODES[c[1]] for c in cols])
    pool = np.array([1 if meta.loc[c, "kind"] == "pool" else 0 for c in cols])
    out = []
    for o in origins:
        if o not in idx:
            continue
        io = idx[o]
        if io < 27:
            continue
        hist = Y[:io]  # up to o-1
        L = hist[-12:].mean(axis=0)
        Ls = np.where(L > 1e-9, L, np.nan)
        lag1, lag2, lag3 = hist[-1], hist[-2], hist[-3]
        mean3, mean6 = hist[-3:].mean(0), hist[-6:].mean(0)
        std12 = hist[-12:].std(0)
        prev3 = hist[-15:-12].mean(0)
        yoy3 = np.where(prev3 > 1e-9, mean3 / np.where(prev3 > 1e-9, prev3, 1), 1.0)
        L_prev = hist[-24:-12].mean(0)
        trend12 = np.where(L_prev > 1e-9, L / np.where(L_prev > 1e-9, L_prev, 1), 1.0)
        for h in horizons:
            it = io + h - 1
            t = o + pd.DateOffset(months=h - 1)
            s12 = Y[it - 12] if it - 12 < io else np.full(Y.shape[1], np.nan)
            s24 = Y[it - 24]
            s12p = Y[it - 13]
            s12n = Y[it - 11] if it - 11 < io else s12
            row = pd.DataFrame({
                "region": [c[0] for c in cols], "product_name": [c[1] for c in cols],
                "origin": o, "target_month": t, "h": h,
                "month_t": t.month, "sin_t": np.sin(2 * np.pi * t.month / 12),
                "cos_t": np.cos(2 * np.pi * t.month / 12),
                "product": prod, "area": area, "is_pool": pool,
                "level": L, "log_level": np.log1p(L),
                "r_lag1": lag1 / Ls, "r_lag2": lag2 / Ls, "r_lag3": lag3 / Ls,
                "r_seas12": s12 / Ls, "r_seas24": s24 / Ls, "r_seas12_prev": s12p / Ls,
                "r_seas12_next": s12n / Ls, "r_mean3": mean3 / Ls, "r_mean6": mean6 / Ls,
                "r_std12": std12 / Ls, "yoy3": yoy3, "trend12": trend12,
                "seas12_raw": s12, "mean3_raw": mean3, "prev3_raw": prev3,
            })
            if with_target and it < len(Y):
                row["y"] = Y[it]
            else:
                row["y"] = np.nan
            out.append(row)
    rows = pd.concat(out, ignore_index=True)
    rows["r_y"] = rows["y"] / rows["level"].where(rows["level"] > 1e-9)
    return rows


@dataclass
class QuantileDemandForecaster:
    params: Dict = field(default_factory=lambda: dict(
        n_estimators=1500, max_depth=5, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
        tree_method="hist", random_state=42, n_jobs=2))
    early_stopping_rounds: int = 60
    calib_months: int = 24
    models: Dict[float, xgb.XGBRegressor] = field(default_factory=dict)
    best_iter: Dict[float, int] = field(default_factory=dict)
    conformal_q: Dict[int, float] = field(default_factory=dict)  # per horizon, ratio units
    train_end: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------ #
    def _fit_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        r = rows.dropna(subset=["r_y"])
        r = r[r["level"] >= MIN_LEVEL_KT]
        return r[np.isfinite(r["r_y"])]

    def fit(self, panel: pd.DataFrame, train_end: pd.Timestamp,
            first_origin: pd.Timestamp = pd.Timestamp("1995-04-01")) -> "QuantileDemandForecaster":
        """Fit using only targets up to `train_end` (inclusive).

        The last `calib_months` of targets are held out for early stopping and
        conformal calibration; the model is then refit on everything with the
        selected number of trees.
        """
        self.train_end = pd.Timestamp(train_end)
        calib_start = self.train_end - pd.DateOffset(months=self.calib_months - 1)
        origins = pd.date_range(first_origin, self.train_end, freq="MS")
        rows = build_rows(panel, origins)
        rows = self._fit_rows(rows[rows["target_month"] <= self.train_end])
        tr = rows[rows["target_month"] < calib_start]
        # calibration rows: origins inside the window whose targets are also inside
        ca = rows[(rows["origin"] >= calib_start)]
        X_tr, X_ca = tr[FEATURES], ca[FEATURES]
        for q in QUANTILES:
            m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                                 early_stopping_rounds=self.early_stopping_rounds, **self.params)
            m.fit(X_tr, tr["r_y"], sample_weight=tr["level"],
                  eval_set=[(X_ca, ca["r_y"])], sample_weight_eval_set=[ca["level"]], verbose=False)
            self.best_iter[q] = int(m.best_iteration) + 1
            self.models[q] = m
        # conformal (CQR) on the calibration window, per horizon
        pr = self._predict_ratio(X_ca)
        for h in HORIZONS:
            mask = (ca["h"] == h).to_numpy()
            lo, hi, y = pr[0.1][mask], pr[0.9][mask], ca["r_y"].to_numpy()[mask]
            score = np.maximum(lo - y, y - hi)
            n = len(score)
            level = min(1.0, np.ceil((n + 1) * 0.8) / n)
            self.conformal_q[h] = float(np.quantile(score, level, method="higher"))
        # refit on all rows with the selected number of trees
        params = dict(self.params)
        for q in QUANTILES:
            params["n_estimators"] = max(50, self.best_iter[q])
            m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **params)
            m.fit(rows[FEATURES], rows["r_y"], sample_weight=rows["level"], verbose=False)
            self.models[q] = m
        return self

    def _predict_ratio(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        return {q: self.models[q].predict(X) for q in QUANTILES}

    def predict_rows(self, rows: pd.DataFrame, conformal: bool = True) -> pd.DataFrame:
        out = rows[["region", "product_name", "origin", "target_month", "h", "level", "y",
                    "seas12_raw", "mean3_raw", "prev3_raw"]].copy()
        pr = self._predict_ratio(rows[FEATURES])
        lo, med, hi = pr[0.1], pr[0.5], pr[0.9]
        if conformal and self.conformal_q:
            adj = rows["h"].map(self.conformal_q).to_numpy()
            lo, hi = lo - adj, hi + adj
        stack = np.sort(np.vstack([lo, med, hi]), axis=0)
        L = rows["level"].to_numpy()
        small = L < MIN_LEVEL_KT
        for i, name in enumerate(["P10", "P50", "P90"]):
            v = np.clip(stack[i] * L, 0, None)
            v[small] = np.nan_to_num(rows["mean3_raw"].to_numpy()[small])
            out[name] = v
        return out

    def forecast(self, panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        rows = build_rows(panel, [pd.Timestamp(origin)], with_target=True)
        return self.predict_rows(rows)


# --------------------------------------------------------------------------- #
# Baselines and metrics
# --------------------------------------------------------------------------- #
def baseline_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["region", "product_name", "origin", "target_month", "h", "y"]].copy()
    out["snaive"] = rows["seas12_raw"].fillna(rows["mean3_raw"])
    growth = (rows["mean3_raw"] / rows["prev3_raw"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["snaive_growth"] = (rows["seas12_raw"] * growth.clip(0.5, 2.0)).fillna(rows["mean3_raw"])
    return out


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def forecast_metrics(df: pd.DataFrame, point_col: str = "P50", with_intervals: bool = True) -> Dict:
    d = df.dropna(subset=["y", point_col])
    y, p = d["y"].to_numpy(), d[point_col].to_numpy()
    out = {"n": int(len(d)), "WAPE": float(np.abs(y - p).sum() / y.sum()),
           "bias_pct": float((p - y).sum() / y.sum() * 100), "MAE_kt": float(np.abs(y - p).mean())}
    if with_intervals and {"P10", "P90"} <= set(d.columns):
        lo, hi = d["P10"].to_numpy(), d["P90"].to_numpy()
        out["coverage_80"] = float(((y >= lo) & (y <= hi)).mean())
        w = d["level"].to_numpy() if "level" in d else np.ones_like(y)
        out["coverage_80_volume_weighted"] = float((((y >= lo) & (y <= hi)) * y).sum() / y.sum())
        out["rel_width"] = float((hi - lo).sum() / y.sum())
        pl = (pinball(y, lo, 0.1) + pinball(y, p, 0.5) + pinball(y, hi, 0.9)) / 3
        out["mean_pinball_scaled"] = float(pl.sum() / y.sum())
    return out
