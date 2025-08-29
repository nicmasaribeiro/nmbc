# finance_lab.py
# Drop in your project root (same level as app.py) or a /services directory.

from __future__ import annotations
import math
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============ Utilities ============

def _as_datetime_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    if date_col and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to coerce index
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(a, b)
        out[~np.isfinite(out)] = 0.0
    return out


# ============ Yield Curve (no SciPy) ============

class YieldCurve:
    """
    Simple piecewise-linear curve on (tenor_years -> annualized rate).
    Provide a dict like {"1M": 0.05, "6M":0.052, "1Y":0.053, "2Y":0.054, "10Y":0.057}
    Rates are continuously compounded if cc=True; otherwise treated as simple APR with compounding='annual'.
    """
    TENOR_MAP = {
        "1W": 7/365, "2W": 14/365, "3W": 21/365,
        "1M": 1/12, "2M": 2/12, "3M": 3/12, "6M": 0.5, "9M": 0.75,
        "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0
    }

    def __init__(self, points: Dict[str, float], cc: bool = False, compounding: str = "annual"):
        pairs = []
        for k, v in points.items():
            t = self.to_years(k)
            pairs.append((t, float(v)))
        pairs = sorted(pairs)
        self.ts = np.array([p[0] for p in pairs], dtype=float)
        self.rs = np.array([p[1] for p in pairs], dtype=float)
        self.cc = cc
        self.compounding = compounding

    @classmethod
    def to_years(cls, tenor: str) -> float:
        tenor = tenor.upper().strip()
        if tenor in cls.TENOR_MAP:
            return cls.TENOR_MAP[tenor]
        if tenor.endswith("Y"):
            return float(tenor[:-1])
        if tenor.endswith("M"):
            return float(tenor[:-1]) / 12.0
        if tenor.endswith("W"):
            return float(tenor[:-1]) * 7/365
        if tenor.endswith("D"):
            return float(tenor[:-1]) / 365.0
        raise ValueError(f"Unrecognized tenor: {tenor}")

    def spot(self, T: float) -> float:
        """Annualized spot rate for maturity T (years) via linear interpolation."""
        if T <= self.ts[0]:
            return self.rs[0]
        if T >= self.ts[-1]:
            return self.rs[-1]
        return float(np.interp(T, self.ts, self.rs))

    def df(self, T: float) -> float:
        """Discount factor for T with current compounding convention."""
        r = self.spot(T)
        if self.cc:
            return math.exp(-r * T)
        if self.compounding == "annual":
            return (1.0 + r) ** (-T)
        if self.compounding == "semi":
            return (1.0 + r/2) ** (-2*T)
        # default fallback
        return (1.0 + r) ** (-T)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simple forward rate implied by DFs between t1 and t2 (annualized)."""
        if t2 <= t1:
            return 0.0
        d1, d2 = self.df(t1), self.df(t2)
        # Simple-comp forward approximation
        f = (d1/d2 - 1.0) / (t2 - t1)
        return float(f)


# ============ Risk-Neutral Pricing ============

@dataclass
class BSResult:
    price: float
    d1: float
    d2: float
    call: bool

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x):
    return (1.0 / math.sqrt(2.0*math.pi)) * math.exp(-0.5 * x*x)

def black_scholes(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True) -> BSResult:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # trivial fallback
        intrinsic = max(S - K, 0.0) if call else max(K - S, 0.0)
        return BSResult(price=float(intrinsic), d1=0.0, d2=0.0, call=call)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if call:
        price = S*math.exp(-q*T)*_norm_cdf(d1) - K*math.exp(-r*T)*_norm_cdf(d2)
    else:
        price = K*math.exp(-r*T)*_norm_cdf(-d2) - S*math.exp(-q*T)*_norm_cdf(-d1)
    return BSResult(price=float(price), d1=d1, d2=d2, call=call)

def risk_neutral_mc_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
                          call: bool = True, steps: int = 252, n_paths: int = 10000, seed: Optional[int] = None) -> float:
    rng = np.random.default_rng(seed)
    dt = T/steps
    drift = (r - q - 0.5*sigma*sigma)*dt
    vol = sigma * math.sqrt(dt)
    z = rng.standard_normal((n_paths, steps))
    log_path = np.log(S) + np.cumsum(drift + vol*z, axis=1)
    ST = np.exp(log_path[:, -1])
    payoff = np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)
    return float(np.exp(-r*T) * payoff.mean())


# ============ Feature Engineering ============

class FeatureEngineer:
    def __init__(self, close_col: str = "Close", volume_col: Optional[str] = None):
        self.close_col = close_col
        self.volume_col = volume_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        price = df[self.close_col].astype(float)

        # Core series
        df["log_ret_1"] = np.log(price).diff()
        df["ret_1"] = price.pct_change()
        df["ret_5"] = price.pct_change(5)
        df["ret_21"] = price.pct_change(21)
        df["mom_21"] = price / price.shift(21) - 1.0

        # Moving averages & vol
        df["sma_10"] = price.rolling(10).mean()
        df["sma_20"] = price.rolling(20).mean()
        df["ema_12"] = price.ewm(span=12, adjust=False).mean()
        df["ema_26"] = price.ewm(span=26, adjust=False).mean()
        df["vol_21"] = df["ret_1"].rolling(21).std()
        df["bb_width_20"] = (price.rolling(20).mean() + 2*price.rolling(20).std()) - (price.rolling(20).mean() - 2*price.rolling(20).std())

        # RSI
        delta = price.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = _safe_div(roll_up, roll_down)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        macd = df["ema_12"] - df["ema_26"]
        df["macd"] = macd
        df["macd_sig"] = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_sig"]

        # Optional OBV if volume exists
        if self.volume_col and self.volume_col in df.columns:
            vol = df[self.volume_col].fillna(0).astype(float)
            sign = np.sign(df["ret_1"]).fillna(0.0)
            df["obv"] = (sign * vol).cumsum()

        return df

    @staticmethod
    def make_target(df: pd.DataFrame, close_col: str = "Close", horizon_days: int = 21, log_return: bool = True) -> pd.Series:
        price = df[close_col].astype(float)
        if log_return:
            y = np.log(price.shift(-horizon_days)) - np.log(price)
        else:
            y = price.shift(-horizon_days) / price - 1.0
        return y.rename(f"target_{'log' if log_return else 'pct'}_{horizon_days}d")


# ============ Simple Scaler & Models (NumPy only) ============

class StandardScalerNP:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class LinearRegressionNP:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            X_ = np.c_[np.ones(len(X)), X]
        else:
            X_ = X
        beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        else:
            self.coef_ = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = X @ self.coef_
        if self.fit_intercept:
            y = y + self.intercept_
        return y

class RidgeRegressionNP:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        if self.fit_intercept:
            X_ = np.c_[np.ones(n), X]
        else:
            X_ = X
        # Penalize all but intercept:
        reg = self.alpha * np.eye(X_.shape[1])
        if self.fit_intercept:
            reg[0, 0] = 0.0
        A = X_.T @ X_ + reg
        b = X_.T @ y
        beta = np.linalg.solve(A, b)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        else:
            self.coef_ = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = X @ self.coef_
        if self.fit_intercept:
            y = y + self.intercept_
        return y


# ============ Walk-Forward Backtest ============

@dataclass
class BacktestResult:
    equity: pd.Series
    daily_ret: pd.Series
    trades: pd.Series
    preds: pd.Series
    realized: pd.Series
    metrics: Dict[str, float]
    diagnostics: Dict[str, float]
    csv_bytes: bytes

def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    train_min_obs: int = 252*2,
    retrain_every: int = 21,
    model: str = "ridge",
    ridge_alpha: float = 1.0,
    standardize: bool = True,
) -> pd.Series:
    X_np = X.values.astype(float)
    y_np = y.values.astype(float)

    scaler = StandardScalerNP() if standardize else None
    if scaler:
        X_scaled = scaler.fit_transform(X_np[:train_min_obs])
    preds = np.full_like(y_np, fill_value=np.nan)

    i = train_min_obs
    last_fit_i = None
    reg = RidgeRegressionNP(alpha=ridge_alpha) if model == "ridge" else LinearRegressionNP()

    while i < len(X_np):
        # retrain if needed
        if last_fit_i is None or (i - last_fit_i) >= retrain_every:
            X_tr = X_np[:i]
            y_tr = y_np[:i]
            if scaler:
                scaler.fit(X_tr)
                X_fit = scaler.transform(X_tr)
            else:
                X_fit = X_tr
            reg.fit(X_fit, y_tr)
            last_fit_i = i
        # one-step-ahead prediction
        X_te = X_np[i:i+1]
        X_te = scaler.transform(X_te) if scaler else X_te
        preds[i] = reg.predict(X_te)[0]
        i += 1

    return pd.Series(preds, index=X.index, name="pred")


def backtest_from_preds(
    prices: pd.Series,
    preds: pd.Series,
    target: pd.Series,
    tc_bps: float = 1.0,
    leverage: float = 1.0,
) -> BacktestResult:
    """
    prices: close prices series (DatetimeIndex)
    preds: predicted (log) return over horizon h (already no-lookahead)
    target: realized (log) return over same horizon (shifted)
    Strategy: daily sign(pred) position on next day; PnL = pos * next_day_return - costs on pos change.
    """
    prices = prices.astype(float)
    daily_log_ret = np.log(prices).diff().fillna(0.0)

    # Convert horizon prediction into *directional* daily signal
    signal = np.sign(preds).shift(1).fillna(0.0)  # act next day
    signal = signal.clip(-1, 1) * leverage

    # Trading costs on position change (bps of notional)
    pos_change = (signal - signal.shift(1)).fillna(signal)
    costs = (abs(pos_change) * (tc_bps / 10000.0))

    strat_daily = signal * daily_log_ret - costs
    equity = strat_daily.cumsum().apply(np.exp)  # start at 1.0 approx
    equity.iloc[0] = 1.0

    # Metrics
    ann_factor = 252
    mu = strat_daily.mean() * ann_factor
    sigma = strat_daily.std(ddof=0) * math.sqrt(ann_factor)
    sharpe = mu / sigma if sigma > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1.0).min())
    cagr = float((equity.dropna().iloc[-1]) ** (ann_factor/len(equity)) - 1.0) if len(equity) > 0 else 0.0
    hit = float((np.sign(preds) == np.sign(target)).mean())

    metrics = {
        "AnnReturn": float(mu),
        "AnnVol": float(sigma),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(mdd),
        "CAGR": float(cagr),
        "DirectionHitRate": float(hit),
    }
    diagnostics = {
        "MeanPred": float(np.nanmean(preds.values)),
        "StdPred": float(np.nanstd(preds.values)),
        "MeanDailyLogRet": float(daily_log_ret.mean()),
        "TradesPerYear": float((pos_change.abs() > 1e-9).sum() * (ann_factor/len(equity))) if len(equity) else 0.0,
    }

    # CSV export
    out = pd.DataFrame({
        "price": prices,
        "daily_log_ret": daily_log_ret,
        "pred": preds,
        "signal_lagged": signal,
        "pnl_daily_log": strat_daily,
        "equity": equity,
        "realized_target": target,
    })
    buf = io.BytesIO()
    out.to_csv(buf)
    buf.seek(0)

    return BacktestResult(
        equity=equity, daily_ret=strat_daily, trades=pos_change, preds=preds,
        realized=target, metrics=metrics, diagnostics=diagnostics, csv_bytes=buf.read()
    )


# ============ End-to-end convenience ============

def run_lab_pipeline(
    df_prices: pd.DataFrame,
    date_col: Optional[str] = None,
    close_col: str = "Close",
    volume_col: Optional[str] = None,
    horizon_days: int = 21,
    model: str = "ridge",
    ridge_alpha: float = 10.0,
    standardize: bool = True,
    train_min_obs: int = 252*2,
    retrain_every: int = 21,
) -> Tuple[BacktestResult, pd.DataFrame]:
    """
    Returns (BacktestResult, features_df) where features_df includes engineered features and targets.
    """
    df = _as_datetime_index(df_prices, date_col)
    assert close_col in df.columns, f"Missing column '{close_col}'"
    fe = FeatureEngineer(close_col=close_col, volume_col=volume_col)
    feat = fe.transform(df)
    target = fe.make_target(df, close_col=close_col, horizon_days=horizon_days, log_return=True)

    # Build X
    feature_cols = [
        "log_ret_1","ret_1","ret_5","ret_21","mom_21","sma_10","sma_20",
        "ema_12","ema_26","vol_21","bb_width_20","rsi_14","macd","macd_sig","macd_hist"
    ]
    feature_cols = [c for c in feature_cols if c in feat.columns]
    X = feat[feature_cols].dropna()
    y = target.reindex(X.index)

    # Ensure no NaN in y
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    preds = walk_forward_predict(
        X=X, y=y,
        train_min_obs=train_min_obs,
        retrain_every=retrain_every,
        model=model, ridge_alpha=ridge_alpha,
        standardize=standardize
    )

    res = backtest_from_preds(
        prices=df[close_col].reindex(preds.index).ffill(),
        preds=preds,
        target=y,
        tc_bps=1.0,
        leverage=1.0
    )
    # Pack features for debug/inspection
    features_df = feat.reindex(preds.index).assign(target=y, pred=preds)
    return res, features_df
