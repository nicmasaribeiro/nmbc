"""
opt_models.py — SQLAlchemy model for storing optimization results.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from models import db  # shared SQLAlchemy instance


class OptimizationResult(db.Model):
    __tablename__ = "optimization_result"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # assets in order used by optimizer
    assets = db.Column(db.JSON, nullable=False)   # List[str]
    weights = db.Column(db.JSON, nullable=False)  # Dict[str, float]

    expected_return = db.Column(db.Float, nullable=True)
    expected_volatility = db.Column(db.Float, nullable=True)
    sharpe = db.Column(db.Float, nullable=True)

    constraints = db.Column(db.JSON, nullable=True)  # dict of params
    method = db.Column(db.String(32), nullable=True)
    lookback_days = db.Column(db.Integer, nullable=True)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "assets": self.assets,
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe": self.sharpe,
            "constraints": self.constraints,
            "method": self.method,
            "lookback_days": self.lookback_days,
        }

    def __repr__(self) -> str:
        return f"<OptimizationResult id={self.id} user_id={self.user_id} method={self.method}>"