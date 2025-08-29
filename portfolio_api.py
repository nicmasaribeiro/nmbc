"""
portfolio_api.py — Flask blueprint exposing /portfolio/optimize
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify
from flask_login import current_user

from models import db  # shared SQLAlchemy session
from optimizer import run_markowitz_optimization, explain_optimization

portfolio_bp = Blueprint("portfolio", __name__)


@portfolio_bp.route("/portfolio/optimize", methods=["POST"])
def optimize_portfolio():
    payload = request.get_json(silent=True) or {}

    user_id = payload.get("user_id")
    if not user_id and hasattr(current_user, "id"):
        try:
            user_id = int(current_user.id)
        except Exception:
            user_id = None

    target_return = payload.get("target_return")  # float or None
    risk_free_rate = float(payload.get("risk_free_rate", 0.0))
    lookback_days = int(payload.get("lookback_days", 252))
    allow_short = bool(payload.get("allow_short", False))
    frontier_points = int(payload.get("frontier_points", 0))
    explain = bool(payload.get("explain", True))
    store = bool(payload.get("store", True))

    try:
        result = run_markowitz_optimization(
            session=db.session,
            user_id=user_id,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
            lookback_days=lookback_days,
            allow_short=allow_short,
            frontier_points=frontier_points,
            store_in_db=store,
        )
        resp = {
            "assets": result.assets,
            "weights": result.weights,
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "sharpe": result.sharpe,
        }
        if result.frontier:
            resp["frontier"] = result.frontier
        if result.notes:
            resp["notes"] = result.notes
        if explain:
            resp["explanation"] = explain_optimization(result)
        return jsonify(resp), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400