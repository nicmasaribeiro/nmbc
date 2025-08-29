# datasets_blueprint.py
import os, io, hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, request, render_template, redirect, url_for, flash,
    send_file, abort, jsonify, current_app
)
from flask_login import login_required, current_user
from sqlalchemy import or_, and_
import mimetypes

from models import db, Dataset, DatasetVersion, DatasetPurchase

# Optional: pandas for CSV/Parquet preview
try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

datasets = Blueprint("datasets", __name__, template_folder="templates")

ALLOWED_EXT = {".csv", ".tsv", ".parquet", ".jsonl", ".json", ".xlsx", ".zip"}
DATASET_DIR = "uploads/datasets"  # ensure this exists and is writable

def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def sha256_file(fp) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fp.read(8192), b""):
        h.update(chunk)
    fp.seek(0)
    return h.hexdigest()

def parse_tags(s: str) -> str:
    # Normalize tags to comma-separated, unique, lowercase
    tags = [t.strip().lower() for t in s.split(",") if t.strip()]
    return ",".join(sorted(set(tags)))

def can_download(user_id: int, d: Dataset) -> bool:
    if d.visibility == "public" and d.price_cents == 0:
        return True
    if user_id and d.owner_id == user_id:
        return True
    if d.price_cents == 0:  # free but private -> owner only
        return d.visibility == "public"
    # check purchase
    if not user_id:
        return False
    return db.session.query(DatasetPurchase.id).filter_by(dataset_id=d.id, buyer_id=user_id).first() is not None

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def version_str(existing_versions):
    # v1, v2, ...
    if not existing_versions:
        return "v1"
    latest = max(
        [int(v.version[1:]) for v in existing_versions if v.version.startswith("v") and v.version[1:].isdigit()],
        default=0
    )
    return f"v{latest+1}"

def preview_dataframe(filepath: str, max_rows=15):
    if not HAVE_PANDAS:
        return 0, 0, []
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(filepath, sep=sep, nrows=max_rows)
        elif ext == ".parquet":
            df = pd.read_parquet(filepath).head(max_rows)
        elif ext == ".jsonl":
            df = pd.read_json(filepath, lines=True, nrows=max_rows)
        elif ext == ".json":
            df = pd.read_json(filepath)
            if len(df) > max_rows:
                df = df.head(max_rows)
        elif ext == ".xlsx":
            df = pd.read_excel(filepath, nrows=max_rows)
        else:
            return 0, 0, []
        rows, cols = df.shape
        sample = df.astype(object).fillna("").to_dict(orient="records")
        return rows, cols, sample
    except Exception:
        return 0, 0, []

@datasets.route("/datasets", methods=["GET"])
def list_datasets():
    q = request.args.get("q", "").strip()
    tag = request.args.get("tag", "").strip().lower()
    owner = request.args.get("owner", "").strip()
    v = request.args.get("visibility", "").strip().lower()
    min_price = request.args.get("min_price", type=int)
    max_price = request.args.get("max_price", type=int)

    qry = Dataset.query
    if q:
        like = f"%{q}%"
        qry = qry.filter(or_(Dataset.title.ilike(like), Dataset.description.ilike(like), Dataset.tags.ilike(like)))
    if tag:
        qry = qry.filter(Dataset.tags.ilike(f"%{tag}%"))
    if owner and owner.isdigit():
        qry = qry.filter(Dataset.owner_id == int(owner))
    if v in {"public", "private"}:
        qry = qry.filter(Dataset.visibility == v)
    if min_price is not None:
        qry = qry.filter(Dataset.price_cents >= min_price)
    if max_price is not None:
        qry = qry.filter(Dataset.price_cents <= max_price)

    qry = qry.order_by(Dataset.updated_at.desc())
    items = qry.limit(50).all()

    # HTML or JSON
    if request.accept_mimetypes.best == "application/json" or request.args.get("format") == "json":
        return jsonify([
            {
                "id": d.id, "title": d.title, "description": d.description[:300],
                "tags": d.tag_list(), "price_cents": d.price_cents,
                "visibility": d.visibility, "owner_id": d.owner_id,
                "downloads": d.downloads, "purchases": d.purchases,
                "created_at": d.created_at.isoformat(), "updated_at": d.updated_at.isoformat()
            } for d in items
        ])

    return render_template("datasets_list.html", items=items, q=q, tag=tag)

@datasets.route("/datasets/<int:dataset_id>", methods=["GET"])
def dataset_detail(dataset_id):
    d = Dataset.query.get_or_404(dataset_id)
    # for preview, use latest version
    ver = d.latest_version()
    sample = []
    if ver and ver.sample_json:
        try:
            sample = json.loads(ver.sample_json)
        except Exception:
            sample = []

    if request.accept_mimetypes.best == "application/json" or request.args.get("format") == "json":
        return jsonify({
            "id": d.id, "title": d.title, "description": d.description,
            "tags": d.tag_list(), "price_cents": d.price_cents, "license": d.license,
            "visibility": d.visibility, "owner_id": d.owner_id,
            "downloads": d.downloads, "purchases": d.purchases,
            "versions": [{
                "id": v.id, "version": v.version, "filename": v.filename,
                "size": v.file_size, "mime": v.file_mime, "rows": v.rows,
                "cols": v.cols, "created_at": v.created_at.isoformat()
            } for v in d.versions],
            "preview_sample": sample[:15],
        })

    return render_template("dataset_detail.html", d=d, sample=sample[:15])

import json

@datasets.route("/datasets/new", methods=["GET", "POST"])
@login_required
def dataset_new():
    if request.method == "GET":
        return render_template("dataset_new.html")

    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    tags = parse_tags(request.form.get("tags") or "")
    license_ = (request.form.get("license") or "CC-BY-4.0").strip()
    visibility = (request.form.get("visibility") or "public").strip().lower()
    price_cents = int(request.form.get("price_cents") or 0)

    f = request.files.get("file")
    if not title or not f:
        flash("Title and file are required.", "error")
        return redirect(url_for("datasets.dataset_new"))

    filename = secure_filename(f.filename)
    if not allowed_file(filename):
        flash("Unsupported file type.", "error")
        return redirect(url_for("datasets.dataset_new"))

    d = Dataset(
        owner_id=current_user.id,
        title=title, description=description, tags=tags,
        license=license_, visibility="public" if visibility!="private" else "private",
        price_cents=max(0, price_cents),
    )
    db.session.add(d)
    db.session.flush()  # get id

    ver_label = "v1"
    # store file
    base_dir = os.path.join(current_app.instance_path, DATASET_DIR, str(d.id), ver_label)
    ensure_dirs(base_dir)
    path = os.path.join(base_dir, filename)
    f.save(path)

    # compute metadata
    with open(path, "rb") as fp:
        digest = sha256_file(fp)
    sz = os.path.getsize(path)
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

    rows, cols, sample = preview_dataframe(path)
    v = DatasetVersion(
        dataset_id=d.id, version=ver_label, filename=filename,
        file_size=sz, file_mime=mime, sha256=digest,
        rows=rows, cols=cols, sample_json=json.dumps(sample) if sample else "[]"
    )
    db.session.add(v)
    db.session.commit()

    flash("Dataset created.", "success")
    return redirect(url_for("datasets.dataset_detail", dataset_id=d.id))

@datasets.route("/datasets/<int:dataset_id>/upload", methods=["POST"])
@login_required
def dataset_upload_version(dataset_id):
    d = Dataset.query.get_or_404(dataset_id)
    if d.owner_id != current_user.id:
        abort(403)

    f = request.files.get("file")
    if not f:
        flash("Select a file to upload.", "error")
        return redirect(url_for("datasets.dataset_detail", dataset_id=dataset_id))

    filename = secure_filename(f.filename)
    if not allowed_file(filename):
        flash("Unsupported file type.", "error")
        return redirect(url_for("datasets.dataset_detail", dataset_id=dataset_id))

    ver_label = version_str(d.versions)
    base_dir = os.path.join(current_app.instance_path, DATASET_DIR, str(d.id), ver_label)
    ensure_dirs(base_dir)
    path = os.path.join(base_dir, filename)
    f.save(path)

    with open(path, "rb") as fp:
        digest = sha256_file(fp)
    sz = os.path.getsize(path)
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    rows, cols, sample = preview_dataframe(path)

    v = DatasetVersion(
        dataset_id=d.id, version=ver_label, filename=filename,
        file_size=sz, file_mime=mime, sha256=digest,
        rows=rows, cols=cols, sample_json=json.dumps(sample) if sample else "[]"
    )
    db.session.add(v)
    db.session.commit()

    flash(f"Uploaded {ver_label}.", "success")
    return redirect(url_for("datasets.dataset_detail", dataset_id=d.id))

@datasets.route("/datasets/<int:dataset_id>/edit", methods=["POST"])
@login_required
def dataset_edit(dataset_id):
    d = Dataset.query.get_or_404(dataset_id)
    if d.owner_id != current_user.id:
        abort(403)

    d.title = (request.form.get("title") or d.title).strip()
    d.description = (request.form.get("description") or d.description).strip()
    d.tags = parse_tags(request.form.get("tags") or d.tags)
    d.license = (request.form.get("license") or d.license).strip()
    visibility = (request.form.get("visibility") or d.visibility).strip().lower()
    d.visibility = "public" if visibility!="private" else "private"
    try:
        d.price_cents = max(0, int(request.form.get("price_cents") or d.price_cents))
    except Exception:
        pass

    db.session.commit()
    flash("Dataset updated.", "success")
    return redirect(url_for("datasets.dataset_detail", dataset_id=d.id))

@datasets.route("/datasets/<int:dataset_id>/download", methods=["GET"])
@login_required
def dataset_download(dataset_id):
    d = Dataset.query.get_or_404(dataset_id)
    if not can_download(current_user.id, d):
        abort(403)

    want_version = request.args.get("version")
    v = None
    if want_version:
        v = DatasetVersion.query.filter_by(dataset_id=d.id, version=want_version).first()
        if not v:
            abort(404)
    else:
        v = d.latest_version()
        if not v:
            abort(404)

    path = os.path.join(current_app.instance_path, DATASET_DIR, str(d.id), v.version, v.filename)
    if not os.path.exists(path):
        abort(404)

    d.downloads += 1
    db.session.commit()

    return send_file(path, as_attachment=True, download_name=v.filename, mimetype=v.file_mime)

@datasets.route("/datasets/<int:dataset_id>/buy", methods=["POST"])
@login_required
def dataset_buy(dataset_id):
    # You can wire this into your on-chain treasury/BettingHouse later.
    d = Dataset.query.get_or_404(dataset_id)
    if d.price_cents <= 0:
        flash("This dataset is free.", "info")
        return redirect(url_for("datasets.dataset_detail", dataset_id=dataset_id))

    # TODO: integrate real payment. For now, record a purchase with amount_cents.
    purchase = DatasetPurchase(dataset_id=d.id, buyer_id=current_user.id, amount_cents=d.price_cents)
    d.purchases += 1
    db.session.add(purchase)
    db.session.commit()
    flash("Purchase recorded. You can now download.", "success")
    return redirect(url_for("datasets.dataset_detail", dataset_id=d.id))

@datasets.route("/datasets/<int:dataset_id>/delete", methods=["POST"])
@login_required
def dataset_delete(dataset_id):
    d = Dataset.query.get_or_404(dataset_id)
    if d.owner_id != current_user.id:
        abort(403)
    db.session.delete(d)
    db.session.commit()
    flash("Dataset deleted.", "success")
    return redirect(url_for("datasets.list_datasets"))
