# finance_bp.py
# Register in app.py with:  app.register_blueprint(finance_bp, url_prefix="/finance")

import base64
from flask import Blueprint, render_template, request, jsonify, send_file, abort
import pandas as pd
import io
from datetime import datetime

from finance_lab import run_lab_pipeline, YieldCurve, black_scholes, risk_neutral_mc_price

finance_bp = Blueprint("finance_bp", __name__, template_folder="templates", static_folder="static")

# In-memory store for last result (simple; swap to DB if needed)
_LAST_RESULT = {"csv": None, "when": None}

@finance_bp.route("/lab", methods=["GET"])
def lab_home():
    return render_template("finance/lab.html")

@finance_bp.route("/lab/run", methods=["POST"])
def lab_run():
    """
    Accepts:
      - file: CSV with Date + Close (and optional Volume)
      - horizon_days, model ('ridge'|'linear'), ridge_alpha, standardize, train_min_obs, retrain_every
      - (optional) date_col, close_col, volume_col
    Returns JSON with metrics + chart data and enables /finance/lab/download
    """
    if "file" not in request.files:
        return abort(400, "Upload a CSV file with Date + Close columns.")
    f = request.files["file"]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return abort(400, f"Failed to read CSV: {e}")

    horizon_days = int(request.form.get("horizon_days", 21))
    model = request.form.get("model", "ridge")
    ridge_alpha = float(request.form.get("ridge_alpha", 10.0))
    standardize = request.form.get("standardize", "true").lower() == "true"
    train_min_obs = int(request.form.get("train_min_obs", 504))
    retrain_every = int(request.form.get("retrain_every", 21))
    date_col = request.form.get("date_col", "Date")
    close_col = request.form.get("close_col", "Close")
    volume_col = request.form.get("volume_col") or None

    result, feats = run_lab_pipeline(
        df_prices=df,
        date_col=date_col,
        close_col=close_col,
        volume_col=volume_col,
        horizon_days=horizon_days,
        model=model,
        ridge_alpha=ridge_alpha,
        standardize=standardize,
        train_min_obs=train_min_obs,
        retrain_every=retrain_every,
    )

    # keep CSV for download
    _LAST_RESULT["csv"] = result.csv_bytes
    _LAST_RESULT["when"] = datetime.utcnow().isoformat()

    # pack a light response
    payload = {
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "equity": [{"t": t.isoformat(), "y": float(v)} for t, v in result.equity.dropna().items()],
        "preds_vs_real": [
            {"t": t.isoformat(), "pred": float(p), "real": float(r)}
            for t, p, r in zip(result.preds.index, result.preds.values, result.realized.values)
            if pd.notna(p) and pd.notna(r)
        ],
    }
    return jsonify(payload)

@finance_bp.route("/lab/download", methods=["GET"])
def lab_download():
    if not _LAST_RESULT["csv"]:
        return abort(404, "No recent backtest to download.")
    buf = io.BytesIO(_LAST_RESULT["csv"])
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="backtest_results.csv")

# ---- Risk-neutral pricing helpers (optional UI hooks) ----

@finance_bp.route("/lab/price/bs", methods=["POST"])
def lab_price_bs():
    data = request.get_json(force=True)
    S = float(data["S"]); K = float(data["K"]); T = float(data["T"])
    r = float(data.get("r", 0.05)); q = float(data.get("q", 0.0))
    sigma = float(data["sigma"]); call = bool(data.get("call", True))
    res = black_scholes(S, K, T, r, q, sigma, call)
    return jsonify({"price": res.price, "d1": res.d1, "d2": res.d2})

@finance_bp.route("/lab/price/mc", methods=["POST"])
def lab_price_mc():
    data = request.get_json(force=True)
    S = float(data["S"]); K = float(data["K"]); T = float(data["T"])
    r = float(data.get("r", 0.05)); q = float(data.get("q", 0.0))
    sigma = float(data["sigma"]); call = bool(data.get("call", True))
    steps = int(data.get("steps", 252)); n_paths = int(data.get("n_paths", 5000))
    price = risk_neutral_mc_price(S, K, T, r, q, sigma, call, steps, n_paths)
    return jsonify({"price": price})
