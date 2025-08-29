"""
finance_llm_explainer.py — helper entrypoints that a 'Finance LLM' agent can call.
Keeps dependencies tiny and isolates optimizer invocation + explanation.
"""

from typing import Any, Dict, Tuple

from models import db
from optimizer import run_markowitz_optimization, explain_optimization


def trigger_optimizer_and_explain(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call from your Finance LLM tool/function to run the optimizer and get a friendly explanation.
    Example params:
      {
        "user_id": 123,
        "target_return": 0.10,      # or None
        "risk_free_rate": 0.02,
        "lookback_days": 252,
        "allow_short": False,
        "frontier_points": 21
      }
    """
    result = run_markowitz_optimization(
        session=db.session,
        user_id=params.get("user_id"),
        target_return=params.get("target_return"),
        risk_free_rate=float(params.get("risk_free_rate", 0.0)),
        lookback_days=int(params.get("lookback_days", 252)),
        allow_short=bool(params.get("allow_short", False)),
        frontier_points=int(params.get("frontier_points", 0)),
        store_in_db=bool(params.get("store", True)),
    )

    payload = {
        "result": {
            "assets": result.assets,
            "weights": result.weights,
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "sharpe": result.sharpe,
            "frontier": result.frontier,
            "notes": result.notes,
        },
        "explanation": explain_optimization(result),
    }
    return payload