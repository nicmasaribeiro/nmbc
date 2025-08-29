# game_bp.py
from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, abort, jsonify, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from sqlalchemy import func
import os, csv, math, uuid
from datetime import datetime
from models import db, Users, WalletDB, FinanceChallenge, FinanceSubmission

game_bp = Blueprint("game", __name__, template_folder="templates")

DATA_DIR = "challenge_data"
TRUTH_DIR = "challenge_truth"
SUBMIT_DIR = "challenge_submissions"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRUTH_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR, exist_ok=True)

# ---------- helpers ----------
def _safe_float(x):
    try:
        return float(x)
    except:
        return None

def evaluate_predictions(ground_path, pred_path, metric="rmse"):
    """
    ground_path CSV: id,y_true
    pred_path   CSV: id,y_pred
    Returns (score, n_evaluated)
    """
    gt = {}
    with open(ground_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not {"id", "y_true"} <= set(r.fieldnames or []):
            raise ValueError("Ground truth must have columns: id,y_true")
        for row in r:
            gt[row["id"]] = _safe_float(row["y_true"])

    matched = []
    with open(pred_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not {"id", "y_pred"} <= set(r.fieldnames or []):
            raise ValueError("Predictions must have columns: id,y_pred")
        for row in r:
            k = row["id"]
            if k in gt:
                y = gt[k]
                yhat = _safe_float(row["y_pred"])
                if y is not None and yhat is not None:
                    matched.append((y, yhat))

    if not matched:
        return (float("inf"), 0)

    errors = [abs(y - yhat) for (y, yhat) in matched]
    if metric.lower() == "mae":
        score = sum(errors) / len(errors)
    else:
        # rmse default
        score = math.sqrt(sum(e * e for e in errors) / len(errors))
    return (score, len(matched))

def pay_prize_coins(challenge_id):
    """
    Simple prize: if challenge has prize_coins > 0, pay full prize to best (lowest score) latest leaderboard.
    """
    ch = FinanceChallenge.query.get(challenge_id)
    if not ch or ch.prize_coins <= 0:
        return
    # best submission (lowest score)
    best = (
        FinanceSubmission.query
        .filter_by(challenge_id=challenge_id)
        .filter(FinanceSubmission.score.isnot(None))
        .order_by(FinanceSubmission.score.asc(), FinanceSubmission.created_at.asc())
        .first()
    )
    if not best:
        return

    # credit winner's wallet by username
    winner = Users.query.get(best.user_id)
    if not winner:
        return
    w = WalletDB.query.filter_by(address=winner.username).first()
    if not w:
        return
    w.coins += ch.prize_coins
    db.session.commit()

# ---------- public pages ----------

@game_bp.route("/challenge/new", methods=["GET", "POST"])
@login_required
def create_challenge():
    # ✅ Only allow admins (basic check: username == admin)
    if current_user.username not in {"admin", "root"}:
        flash("Unauthorized", "danger")
        return redirect(url_for("game.challenges"))

    if request.method == "POST":
        title = request.form.get("title")
        slug = request.form.get("slug")
        description = request.form.get("description")
        metric = request.form.get("metric", "rmse")
        prize = float(request.form.get("prize_coins") or 0)
        deadline_raw = request.form.get("deadline")

        deadline = None
        if deadline_raw:
            try:
                deadline = datetime.fromisoformat(deadline_raw)
            except Exception:
                flash("Invalid deadline format. Use YYYY-MM-DD HH:MM", "danger")

        # handle file uploads
        data_file = request.files.get("data_file")
        truth_file = request.files.get("truth_file")

        if not (title and slug and data_file and truth_file):
            flash("Missing required fields", "danger")
            return redirect(url_for("game.create_challenge"))

        data_name = secure_filename(data_file.filename)
        truth_name = secure_filename(truth_file.filename)

        data_file.save(os.path.join(DATA_DIR, data_name))
        truth_file.save(os.path.join(TRUTH_DIR, truth_name))

        # create record
        challenge = FinanceChallenge(
            title=title,
            slug=slug,
            description=description,
            metric=metric,
            prize_coins=prize,
            deadline=deadline,
            data_filename=data_name,
            groundtruth_filename=truth_name,
        )
        db.session.add(challenge)
        db.session.commit()

        flash("✅ Challenge created!", "success")
        return redirect(url_for("game.list_challenges"))

    return render_template("challenge_new.html")

@game_bp.route("/challenges")
def challenges():
    items = FinanceChallenge.query.order_by(FinanceChallenge.created_at.desc()).all()
    return render_template("challenges_index.html", items=items)

@game_bp.route("/challenge/<slug>")
def challenge_detail(slug):
    ch = FinanceChallenge.query.filter_by(slug=slug).first_or_404()
    # leaderboard (top 20)
    leaderboard = (
        FinanceSubmission.query
        .filter_by(challenge_id=ch.id)
        .filter(FinanceSubmission.score.isnot(None))
        .order_by(FinanceSubmission.score.asc(), FinanceSubmission.created_at.asc())
        .limit(20)
        .all()
    )
    return render_template("challenge_detail.html", ch=ch, leaderboard=leaderboard)

@game_bp.route("/challenge/<slug>/download")
def challenge_download(slug):
    ch = FinanceChallenge.query.filter_by(slug=slug).first_or_404()
    return send_from_directory(DATA_DIR, ch.data_filename, as_attachment=True)

@game_bp.route("/challenge/<slug>/play")
@login_required
def challenge_play(slug):
    ch = FinanceChallenge.query.filter_by(slug=slug).first_or_404()
    return render_template("challenge_play.html", ch=ch)

# ---------- submission ----------
@game_bp.route("/challenge/<slug>/submit", methods=["POST"])
@login_required
def challenge_submit(slug):
    ch = FinanceChallenge.query.filter_by(slug=slug).first_or_404()

    if ch.deadline and datetime.utcnow() > ch.deadline:
        flash("Submissions are closed for this challenge.", "warning")
        return redirect(url_for("game.challenge_detail", slug=slug))

    f = request.files.get("predictions")
    if not f or not f.filename.lower().endswith(".csv"):
        flash("Upload a CSV file named predictions with columns id,y_pred.", "danger")
        return redirect(url_for("game.challenge_play", slug=slug))

    filename = secure_filename(f"{current_user.id}_{uuid.uuid4().hex}.csv")
    save_path = os.path.join(SUBMIT_DIR, filename)
    f.save(save_path)

    # evaluate
    try:
        truth_path = os.path.join(TRUTH_DIR, ch.groundtruth_filename)
        score, n = evaluate_predictions(truth_path, save_path, ch.metric)
        detail = f"{ch.metric.upper()} on {n} rows"
        sub = FinanceSubmission(
            challenge_id=ch.id,
            user_id=current_user.id,
            filename=filename,
            score=score,
            score_detail=detail
        )
        db.session.add(sub)
        db.session.commit()
        flash(f"Submission scored {score:.6f} ({detail}).", "success")

        # (Optional) one-shot payout each submit; you might want cron or admin-trigger.
        # pay_prize_coins(ch.id)

    except Exception as e:
        flash(f"Scoring failed: {e}", "danger")

    return redirect(url_for("game.challenge_detail", slug=slug))

# ---------- leaderboard API (optional JSON) ----------
@game_bp.route("/challenge/<slug>/leaderboard.json")
def leaderboard_json(slug):
    ch = FinanceChallenge.query.filter_by(slug=slug).first_or_404()
    rows = (
        db.session.query(
            FinanceSubmission.id,
            FinanceSubmission.score,
            FinanceSubmission.score_detail,
            FinanceSubmission.created_at,
            Users.username
        )
        .join(Users, Users.id == FinanceSubmission.user_id)
        .filter(FinanceSubmission.challenge_id == ch.id)
        .filter(FinanceSubmission.score.isnot(None))
        .order_by(FinanceSubmission.score.asc(), FinanceSubmission.created_at.asc())
        .limit(100)
        .all()
    )
    data = [
        {
            "id": r.id,
            "username": r.username,
            "score": r.score,
            "score_detail": r.score_detail,
            "submitted_at": r.created_at.isoformat()
        } for r in rows
    ]
    return jsonify({"challenge": ch.slug, "leaderboard": data})

# ---------- admin (simple) ----------
@game_bp.route("/admin/new", methods=["GET", "POST"])
@login_required
def admin_new():
    # guard: only allow certain users/role if you have roles
    if current_user.username not in {"admin", "root"}:
        abort(403)

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        slug = request.form.get("slug", "").strip().lower()
        description = request.form.get("description", "")
        metric = request.form.get("metric", "rmse").lower()
        prize = float(request.form.get("prize_coins", "0") or 0)
        deadline = request.form.get("deadline") or None

        data_file = request.files.get("data_file")
        truth_file = request.files.get("truth_file")

        if not (title and slug and data_file and truth_file):
            flash("Missing required fields.", "danger")
            return redirect(url_for("game.admin_new"))

        data_name = secure_filename(data_file.filename)
        truth_name = secure_filename(truth_file.filename)
        data_file.save(os.path.join(DATA_DIR, data_name))
        truth_file.save(os.path.join(TRUTH_DIR, truth_name))

        ch = FinanceChallenge(
            title=title,
            slug=slug,
            description=description,
            metric=metric if metric in {"rmse","mae"} else "rmse",
            prize_coins=prize,
            data_filename=data_name,
            groundtruth_filename=truth_name,
            deadline=datetime.fromisoformat(deadline) if deadline else None
        )
        db.session.add(ch)
        db.session.commit()
        flash("Challenge created!", "success")
        return redirect(url_for("game.challenges"))

    return render_template("challenge_admin_new.html")
