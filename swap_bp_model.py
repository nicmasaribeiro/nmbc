from flask import Blueprint, request, jsonify
from datetime import datetime
from models import db, Swap, SwapRequest, SwapTransaction

swap_bp = Blueprint('swap', __name__)

@swap_bp.route('/swaps/request', methods=['POST'])
def request_swap():
    data = request.json
    new_request = SwapRequest(
        swap_id=data['swap_id'],
        requester=data['requester'],
        counterparty=data['counterparty'],
        notional=data['notional'],
        fixed_rate=data['fixed_rate'],
        floating_rate_spread=data['floating_rate_spread'],
        status="Pending",
        created_at=datetime.utcnow()
    )
    db.session.add(new_request)
    db.session.commit()
    return jsonify({"message": "Swap request created.", "swap_request_id": new_request.id}), 201

@swap_bp.route('/swaps/respond/<int:request_id>', methods=['POST'])
def respond_swap(request_id):
    data = request.json
    swap_request = SwapRequest.query.get_or_404(request_id)
    
    if data['action'] == 'accept':
        swap_request.status = 'Accepted'
        new_swap = Swap(
            notional=swap_request.notional,
            fixed_rate=swap_request.fixed_rate,
            floating_rate_spread=swap_request.floating_rate_spread,
            counterparty_a=swap_request.requester,
            counterparty_b=swap_request.counterparty
        )
        db.session.add(new_swap)
        db.session.commit()
        return jsonify({"message": "Swap request accepted and swap created."}), 200
    
    elif data['action'] == 'counter':
        swap_request.status = 'Counteroffer Pending'
        counter_request = SwapRequest(
            swap_id=swap_request.swap_id,
            requester=swap_request.counterparty,
            counterparty=swap_request.requester,
            notional=data['notional'],
            fixed_rate=data['fixed_rate'],
            floating_rate_spread=data['floating_rate_spread'],
            status='Pending',
            created_at=datetime.utcnow()
        )
        db.session.add(counter_request)
        db.session.commit()
        return jsonify({"message": "Counteroffer submitted.", "swap_request_id": counter_request.id}), 200
    
    elif data['action'] == 'reject':
        swap_request.status = 'Rejected'
        db.session.commit()
        return jsonify({"message": "Swap request rejected."}), 200

    return jsonify({"message": "Invalid action."}), 400

@swap_bp.route('/swaps/pending', methods=['GET'])
def view_pending_swaps():
    pending_swaps = SwapRequest.query.filter(SwapRequest.status == 'Pending').all()
    return jsonify([{ "id": s.id, "requester": s.requester, "counterparty": s.counterparty, "notional": s.notional, "fixed_rate": s.fixed_rate, "status": s.status } for s in pending_swaps])
