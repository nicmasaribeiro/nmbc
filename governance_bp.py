from flask import Blueprint, request, render_template, redirect, url_for, jsonify
from flask_login import login_required, current_user
from models import db, Users, GovernanceProposal, GovernanceVote

governance_bp = Blueprint("governance", __name__)

# ✅ Create a proposal
@governance_bp.route("/governance/propose", methods=["POST"])
@login_required
def propose():
    if not current_user.is_admin():
        return jsonify({"error": "Only admins/governors can create proposals"}), 403

    data = request.json
    proposal = GovernanceProposal(
        title=data.get("title"),
        description=data.get("description"),
        created_by=current_user.id,
        status="Active"
    )
    db.session.add(proposal)
    db.session.commit()
    return jsonify({"message": "Proposal created", "id": proposal.id})

# ✅ Vote on a proposal
@governance_bp.route("/governance/vote/<int:proposal_id>", methods=["POST"])
@login_required
def vote(proposal_id):
    choice = request.json.get("choice", "").lower()
    if choice not in ["yes", "no", "abstain"]:
        return jsonify({"error": "Invalid vote"}), 400

    # Weight = wallet coins
    weight = current_user.wallet.coins if current_user.wallet else 1.0

    vote = GovernanceVote(
        proposal_id=proposal_id,
        voter_id=current_user.id,
        choice=choice,
        weight=weight
    )
    db.session.add(vote)
    db.session.commit()
    return jsonify({"message": "Vote submitted"})

# ✅ Resolve a proposal
@governance_bp.route("/governance/resolve/<int:proposal_id>", methods=["POST"])
@login_required
def resolve(proposal_id):
    if not current_user.is_admin():
        return jsonify({"error": "Only admins can resolve proposals"}), 403

    proposal = GovernanceProposal.query.get_or_404(proposal_id)
    yes_votes = sum(v.weight for v in proposal.votes if v.choice == "yes")
    no_votes = sum(v.weight for v in proposal.votes if v.choice == "no")

    proposal.result = "yes" if yes_votes > no_votes else "no"
    proposal.status = "Resolved"
    db.session.commit()

    return jsonify({"message": f"Proposal resolved: {proposal.result}"})
