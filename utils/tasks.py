# models.py or a tasks.py util
from datetime import datetime
import numpy as np

def update_once():
    """One atomic pass over all objects; safe + idempotent."""
    # --- Recalculate stochastic price ---
    invests = InvestmentDatabase.query.all()
    for i in invests:
        try:
            t = get_price(i.investment_name.upper())
            if t is None or len(t) == 0:
                continue
            price = t.iloc[-1]  # latest
            # Guard: timestamp may be None initially
            prev_ts = i.timestamp or datetime.now()
            now = datetime.now()
            dt_years = (now - prev_ts).total_seconds() / (365.25 * 24 * 3600)

            # Your stochastic pricing function (keep try/except light)
            try:
                s = stoch_price(
                    1/52, i.time_float, i.risk_neutral, i.spread, i.reversion,
                    price, i.target_price
                )
                i.stoch_price = s
            except Exception:
                pass

            # Update market price
            i.market_price = float(price)

            # Update time_float (count down)
            if i.time_float is not None:
                i.time_float = float(i.time_float) - dt_years

            # Update timestamp
            i.timestamp = now

            # Tokenization math (guard division)
            qty = float(i.quantity or 0.0)
            if qty > 0:
                i.tokenized_price = float(i.market_price) / qty
            else:
                i.tokenized_price = 0.0

            # You don’t have a `coins` field—use `coins_value` which exists
            # Assuming the intent is some compounding token metric:
            # coins_value := tokenized_price * (1 + spread) ** time_float
            tf = float(i.time_float or 0.0)
            sp = float(i.spread or 0.0)
            i.coins_value = float(i.tokenized_price) * ((1.0 + sp) ** tf) if i.tokenized_price else 0.0

            # Change value (log return) vs starting_price (guard)
            sp0 = float(i.starting_price or 0.0)
            if sp0 > 0 and price > 0:
                i.change_value = float(np.log(price) - np.log(sp0))

        except Exception as e:
            # Keep loop resilient; log and continue
            print(f"[update_once] {i.investment_name} error: {e}")

    # --- Update portfolio prices ---
    portfolios = Portfolio.query.all()
    for p in portfolios:
        try:
            t = get_price((p.token_name or p.name or "").upper())
            if t is None or len(t) == 0:
                continue
            p.price = float(t.iloc[-1])  # latest
        except Exception as e:
            print(f"[update_once] portfolio {p.name} error: {e}")

    db.session.commit()
