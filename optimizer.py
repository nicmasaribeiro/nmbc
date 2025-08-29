"""
optimizer.py — Markowitz portfolio optimizer
- Pulls tickers from Portfolio and InvestmentDatabase
- Builds expected returns & covariance via numpy / pandas
- Optimizes weights via scipy.optimize.minimize
- Stores results to DB (opt_models.OptimizationResult)
- Optional: updates Portfolio.target_weight if column exists
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# SciPy is used only for the solver; code fails fast with helpful msg if missing
try:
    from scipy.optimize import minimize
except Exception as e:
    minimize = None

# ---- DB Models --------------------------------------------------------------
# Expect a SQLAlchemy setup with a shared `db` instance in models.py
from models import db, Portfolio, InvestmentDatabase  # type: ignore
from opt_models import OptimizationResult  # type: ignore


# ---- Utilities --------------------------------------------------------------
def _fetch_price_history_for(tickers: List[str],
                             period: str = "2y",
                             interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV close prices for tickers. Tries yfinance first.
    Falls back to user-defined helpers if present.
    """
    if not tickers:
        raise ValueError("No tickers provided to fetch price history for.")

    # Unique & clean
    tickers = sorted({t.strip().upper() for t in tickers if t and str(t).strip()})
    if not tickers:
        raise ValueError("All provided tickers were blank/invalid.")

    # Attempt yfinance
    try:
        import yfinance as yf
        df = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.dropna(how="all").dropna(axis=0)  # drop rows with any NaNs
        df = df[tickers]  # enforce column order
        return df
    except Exception:
        pass

    # Fallback: attempt to use a project-defined helper
    try:
        # Expected optional helper: finance_llm.get_price_history(ticker, period, interval)
        from finance_llm import get_price_history  # type: ignore

        frames = []
        for t in tickers:
            s = get_price_history(t, period=period, interval=interval)
            if isinstance(s, pd.DataFrame):
                s = s["Close"] if "Close" in s.columns else s.iloc[:, 0]
            s = s.rename(t)
            frames.append(s)
        df = pd.concat(frames, axis=1).dropna()
        return df
    except Exception as e:
        raise RuntimeError(
            "Could not fetch price history using yfinance or finance_llm.get_price_history. "
            "Please install `yfinance` or provide a compatible helper."
        ) from e


def _annualize_returns(returns: pd.DataFrame, periods_per_year: int = 252) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a returns DataFrame (rows=time, cols=assets), compute:
    - expected returns (mu) annualized
    - covariance matrix (Sigma) annualized using numpy.cov
    """
    if returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets to compute a covariance matrix.")

    # Expected returns: arithmetic mean * periods/year
    mu = returns.mean(axis=0).values * periods_per_year

    # Covariance (sample) using numpy.cov wants variables in rows, observations in columns.
    # Our returns are rows=time, cols=assets => transpose.
    Sigma = np.cov(returns.values.T) * periods_per_year

    return mu, Sigma


@dataclass
class OptimizationOutput:
    assets: List[str]
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe: float
    frontier: Optional[Dict[str, List[float]]] = None
    notes: Optional[str] = None


# ---- Core Optimizer ---------------------------------------------------------
def run_markowitz_optimization(
    session,
    user_id: Optional[int] = None,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0,
    lookback_days: int = 252,
    period: str = "2y",
    interval: str = "1d",
    allow_short: bool = False,
    frontier_points: int = 0,
    store_in_db: bool = True,
) -> OptimizationOutput:
    """
    Solve for either maximum Sharpe or minimum variance for a target return.
    - If target_return is None => maximize Sharpe.
    - Else => minimize variance with return constraint.
    """

    if minimize is None:
        raise RuntimeError("scipy is required. Please `pip install scipy`.")

    # 1) Gather tickers from Portfolio & InvestmentDatabase
    tickers: List[str] = []

    # Portfolio.token_name
    try:
        q1 = session.query(Portfolio.token_name)
        if user_id and hasattr(Portfolio, "user_id"):
            q1 = q1.filter(Portfolio.user_id == user_id)
        tickers += [r[0] for r in q1.all() if r and r[0]]
    except Exception:
        pass

    # InvestmentDatabase.investment_name
    try:
        q2 = session.query(InvestmentDatabase.investment_name)
        if user_id and hasattr(InvestmentDatabase, "user_id"):
            q2 = q2.filter(InvestmentDatabase.user_id == user_id)
        tickers += [r[0] for r in q2.all() if r and r[0]]
    except Exception:
        pass

    tickers = sorted(set([t.strip().upper() for t in tickers if t]))

    if len(tickers) < 2:
        raise ValueError("Need at least two assets across Portfolio/InvestmentDatabase to optimize.")

    # 2) Price history -> returns
    prices = _fetch_price_history_for(tickers, period=period, interval=interval)
    prices = prices.dropna().iloc[-lookback_days:]  # respect lookback window

    rets = prices.pct_change().dropna()
    mu, Sigma = _annualize_returns(rets, periods_per_year=252)

    n = len(tickers)
    if Sigma.shape != (n, n):
        raise ValueError("Covariance matrix shape mismatch.")

    # 3) Optimization setup
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)  # sum(w) = 1
    w0 = np.ones(n) / n

    def _portfolio_stats(w: np.ndarray) -> Tuple[float, float, float]:
        exp_ret = float(np.dot(w, mu))
        vol = float(np.sqrt(np.dot(w, np.dot(Sigma, w))))
        sharpe = (exp_ret - risk_free_rate) / vol if vol > 0 else -np.inf
        return exp_ret, vol, sharpe

    # Objectives
    if target_return is None:
        # Maximize Sharpe => minimize negative Sharpe
        def obj(w):
            exp_ret, vol, sharpe = _portfolio_stats(w)
            return -sharpe
        problem_cons = cons
    else:
        # Minimize variance s.t. expected return >= target_return
        def obj(w):
            # Just variance part:
            return float(np.dot(w, np.dot(Sigma, w)))
        problem_cons = cons + ({"type": "ineq", "fun": lambda w: np.dot(w, mu) - float(target_return)},)

    # Solve
    res = minimize(obj, w0, bounds=bounds, constraints=problem_cons, method="SLSQP", options={"maxiter": 10_000})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w_opt = res.x / np.sum(res.x)  # renormalize
    exp_ret, vol, sharpe = _portfolio_stats(w_opt)
    weights = {t: float(w_opt[i]) for i, t in enumerate(tickers)}

    frontier_payload = None
    if frontier_points and frontier_points > 1:
        # Build a simple efficient frontier by sweeping returns between percentiles
        r_min, r_max = np.percentile(mu, 10), np.percentile(mu, 90)
        targets = np.linspace(r_min, r_max, frontier_points)
        fr_returns, fr_vols = [], []
        for tr in targets:
            res_tr = minimize(
                lambda w: float(np.dot(w, np.dot(Sigma, w))),
                w0, bounds=bounds, constraints=cons + ({"type": "ineq", "fun": lambda w, tr=tr: np.dot(w, mu) - tr},),
                method="SLSQP", options={"maxiter": 5_000}
            )
            if res_tr.success:
                _, v, _ = _portfolio_stats(res_tr.x / np.sum(res_tr.x))
                fr_returns.append(float(tr))
                fr_vols.append(float(v))
        frontier_payload = {"returns": fr_returns, "volatilities": fr_vols}

    # 4) Persist to DB
    if store_in_db:
        rec = OptimizationResult(
            user_id=user_id,
            assets=tickers,
            weights=weights,
            expected_return=float(exp_ret),
            expected_volatility=float(vol),
            sharpe=float(sharpe),
            constraints={
                "target_return": target_return,
                "risk_free_rate": risk_free_rate,
                "allow_short": allow_short,
                "lookback_days": lookback_days,
            },
            method="max_sharpe" if target_return is None else "min_var_target",
            lookback_days=lookback_days,
        )
        session.add(rec)

        # Optional: update Portfolio.target_weight if the column exists
        try:
            for t, w in weights.items():
                row = session.query(Portfolio).filter(Portfolio.token_name == t).first()
                if row is not None and hasattr(row, "target_weight"):
                    setattr(row, "target_weight", float(w))
        except Exception:
            # Non-fatal if schema doesn't have target_weight
            pass

        session.commit()

    notes = None
    if allow_short and any(v < 0 for v in weights.values()):
        notes = "Short positions allowed; some weights are negative."

    return OptimizationOutput(
        assets=tickers,
        weights=weights,
        expected_return=float(exp_ret),
        expected_volatility=float(vol),
        sharpe=float(sharpe),
        frontier=frontier_payload,
        notes=notes,
    )


def explain_optimization(result: OptimizationOutput) -> str:
    """Plain-English explanation suitable for a 'finance LLM' or API response."""
    if not isinstance(result, OptimizationOutput):
        # duck-typed dict
        weights = result.get("weights", {})
        assets = result.get("assets", list(weights.keys()))
        exp_ret = result.get("expected_return")
        vol = result.get("expected_volatility")
        sharpe = result.get("sharpe")
    else:
        weights = result.weights
        assets = result.assets
        exp_ret = result.expected_return
        vol = result.expected_volatility
        sharpe = result.sharpe

    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:5]
    wl = ", ".join([f"{a}: {w:.2%}" for a, w in top])

    text = [
        "📈 Optimization Summary",
        f"- Optimizer selected {len(assets)} assets.",
        f"- Top weights → {wl}",
        f"- Expected annual return ≈ {exp_ret:.2%}",
        f"- Expected annual volatility ≈ {vol:.2%}",
        f"- Expected Sharpe ≈ {sharpe:.2f} (rf=assumed)",
        "",
        "Notes:",
        "• This is a classical Markowitz allocation using sample mean/variance.",
        "• Results are sensitive to the lookback window and outliers.",
        "• Consider adding constraints (sector caps, min/max weights) as needed."
    ]
    return "\n".join(text)