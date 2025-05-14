# finance_llm.py

from flask import Blueprint, request, jsonify, current_app,render_template
from flask_login import login_required, current_user
import openai
from models import InvestmentDatabase, Portfolio, WalletDB
import json

finance_llm_bp = Blueprint("finance_llm", __name__)

# Preload OpenAI key
openai.api_key = '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'
#current_app.config.get("OPENAI_API_KEY") or 'sk-REPLACE-WITH-YOUR-KEY'


@finance_llm_bp.route("/chat/finance", methods=["POST"])
@login_required
def chat_finance():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Fetch user investment context
        investments = InvestmentDatabase.query.filter_by(owner=current_user.username).all()
        portfolio = Portfolio.query.filter_by(username=current_user.username).all()
        wallet = WalletDB.query.filter_by(address=current_user.username).first()

        context_blob = {
            "investments": [
                {
                    "name": i.investment_name,
                    "market_price": i.market_price,
                    "tokenized_price": i.tokenized_price,
                    "change": i.change_value,
                }
                for i in investments
            ],
            "portfolio": [
                {
                    "name": p.name,
                    "price": p.price,
                    "weight": p.weight,
                }
                for p in portfolio
            ],
            "wallet": {
                "balance": wallet.balance if wallet else 0.0,
                "coins": wallet.coins if wallet else 0.0,
            },
        }

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial assistant with knowledge of derivatives, tokenized swaps, user portfolios, and financial metrics. Help users understand their positions, suggest actions, and calculate results."
                },
                {
                    "role": "user",
                    "content": f"{user_message}\n\nContext: {json.dumps(context_blob, indent=2)}"
                }
            ]
        )

        reply = response["choices"][0]["message"]["content"]
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@finance_llm_bp.route("/chat")
@login_required
def finance_chat_page():
    return render_template("finance_llm_chat.html")
