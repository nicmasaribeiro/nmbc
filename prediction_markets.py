# prediction_markets.py
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, abort
from flask_login import login_required, current_user
from models import db, Users, WalletDB, BettingHouse, PredictionMarket, PredictionBet
from datetime import datetime

markets_bp = Blueprint("markets", __name__, template_folder="templates")

HOUSE_ID = 1              # single treasury row
HOUSE_FEE = 0.02          # 2% fee on losing-side pool (configurable)

def get_house():
    house = BettingHouse.query.get(HOUSE_ID)
    if not house:
        house = BettingHouse(id=HOUSE_ID, balance=0.0, coins=1_000_000_000.0)
        db.session.add(house)
        db.session.commit()
    return house

def get_user_wallet(user):
    # You’ve mapped wallet.address to username elsewhere; keep that convention
    w = WalletDB.query.filter_by(address=user.username).first()
    if not w:
        w = WalletDB(address=user.username, balance=0.0, coins=0.0)
        db.session.add(w)
        db.session.commit()
    return w

@markets_bp.route("/markets")
@login_required
def index():
    markets = PredictionMarket.query.order_by(PredictionMarket.created_at.desc()).all()
    return render_template("markets_index.html", markets=markets)

@markets_bp.route("/markets/<int:market_id>")
@login_required
def detail(market_id):
    m = PredictionMarket.query.get_or_404(market_id)
    user_bets = []
    if current_user.is_authenticated:
        user_bets = PredictionBet.query.filter_by(market_id=m.id, user_id=current_user.id).all()
    return render_template("market_detail.html", m=m, user_bets=user_bets)

@markets_bp.route("/markets/create", methods=["POST"])
@login_required
def create():
    question = request.form.get("question", "").strip()
    if not question:
        return "Question is required", 400
    m = PredictionMarket(question=question, creator_id=current_user.id)
    db.session.add(m)
    db.session.commit()
    return redirect(url_for("markets.detail", market_id=m.id))

@markets_bp.route("/markets/<int:market_id>/bet", methods=["POST"])
@login_required
def bet(market_id):
    m = PredictionMarket.query.get_or_404(market_id)
    if m.resolved:
        return "Market already resolved", 400

    side = request.form.get("side", "").lower()
    try:
        amount = float(request.form.get("amount", "0"))
    except:
        return "Invalid amount", 400

    if side not in ("yes", "no"):
        return "Side must be 'yes' or 'no'", 400
    if amount <= 0:
        return "Amount must be positive", 400

    wallet = get_user_wallet(current_user)
    if wallet.coins < amount:
        return "Insufficient coins", 402

    house = get_house()

    # Move coins: user -> house
    wallet.coins -= amount
    house.coins  += amount

    # Record bet & update pool
    b = PredictionBet(market_id=m.id, user_id=current_user.id, side=side, amount=amount)
    if side == "yes":
        m.outcome_yes += amount
    else:
        m.outcome_no  += amount

    db.session.add(b)
    db.session.commit()
    return redirect(url_for("markets.detail", market_id=m.id))

@markets_bp.route("/markets/<int:market_id>/resolve", methods=["POST"])
@login_required
def resolve(market_id):
    """
    Admin/creator action: set result to "yes" or "no", auto-payout winners.
    Payout = user_stake + pro-rata share of losing pool * (1 - HOUSE_FEE).
    Funds flow: house -> winners.
    """
    m = PredictionMarket.query.get_or_404(market_id)
    if m.resolved:
        return "Already resolved", 400

    result = request.form.get("result", "").lower()
    if result not in ("yes", "no"):
        return "result must be 'yes' or 'no'", 400

    house = get_house()

    total_yes = m.outcome_yes
    total_no  = m.outcome_no
    losing_pot = total_no if result == "yes" else total_yes
    winning_pot = total_yes if result == "yes" else total_no
    fee = losing_pot * HOUSE_FEE
    distributable = max(0.0, losing_pot - fee)

    # Pay winners proportionally (and return their principal)
    winners = PredictionBet.query.filter_by(market_id=m.id, side=result, payout_claimed=False).all()

    # Edge cases: if no winners, fee still retained and losing pot stays in house
    if winners and winning_pot > 0:
        for w in winners:
            share = w.amount / winning_pot
            payout = w.amount + distributable * share

            # house -> user wallet
            uw = get_user_wallet(w.user)
            if house.coins < payout:
                # In practice, you’d guard against treasury insolvency or soft-fail here.
                return "Treasury insufficient for payout", 500

            house.coins -= payout
            uw.coins    += payout
            w.payout_claimed = True

    # Mark resolved & collect fee to house.balance (accounting)
    m.resolved = True
    m.result = result
    house.balance += fee  # optional: track fee revenue separately
    db.session.commit()

    return redirect(url_for("markets.detail", market_id=m.id))
