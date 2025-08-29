# docs_blueprint.py
import os
import shutil
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint, request, render_template, redirect, url_for,
    flash, abort, send_file, jsonify, current_app
)
from flask_login import login_required, current_user
from markdown import markdown as md_to_html
import bleach

# --- IMPORTANT: avoid name collisions ---
from models import db, Document as DocModel  # your SQLAlchemy model
# If you have a LaTeX model too, alias it:
# from models import Latex as LatexModel

# If you need pylatex later, alias it so it doesn't shadow your ORM model:
# from pylatex import Document as PyLaTeXDocument

docs_bp = Blueprint("docs", __name__, url_prefix="/docs", template_folder="templates")

# ---------- Storage ----------
def _storage_root() -> str:
    root = current_app.config.get(
        "DOCS_STORAGE_DIR",
        os.path.join(current_app.instance_path, "docs")
    )
    Path(root).mkdir(parents=True, exist_ok=True)
    return root

def _doc_dir(doc_id: int) -> str:
    root = _storage_root()
    p = os.path.join(root, str(doc_id))
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

# ---------- Markdown sanitize/render ----------
ALLOWED_TAGS = bleach.sanitizer.ALLOWED_TAGS.union({
    "p","pre","code","h1","h2","h3","h4","h5","h6","span","div","hr",
    "img","table","thead","tbody","tr","th","td","blockquote","ul","ol","li"
})
ALLOWED_ATTRS = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "img": ["src","alt","title"],
    "a": ["href","title","target","rel"],
    "span": ["class"],
    "div": ["class"],
    "code": ["class"],
    "pre": ["class"],
}
def render_markdown_safe(text: str) -> str:
    raw_html = md_to_html(text or "", extensions=["extra", "tables", "fenced_code"])
    clean_html = bleach.clean(raw_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
    # allow external links to open safely
    clean_html = bleach.linkify(clean_html, callbacks=[bleach.linkifier.DEFAULT_CALLBACK])
    return clean_html

# ---------- LaTeX compile (optional) ----------
def compile_latex_to_pdf(tex_source_path: str, out_dir: str) -> str | None:
    """Compile .tex to PDF using pdflatex if available. Returns pdf path or None."""
    if shutil.which("pdflatex") is None:
        return None
    cmd = [
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
        "-output-directory", out_dir, tex_source_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pdf_name = os.path.splitext(os.path.basename(tex_source_path))[0] + ".pdf"
        return os.path.join(out_dir, pdf_name)
    except subprocess.CalledProcessError:
        return None

# ---------- Utilities ----------
def slugify(s: str) -> str:
    base = (s or "").strip().lower()
    base = "".join(ch if ch.isalnum() or ch in "-_ " else "-" for ch in base)
    base = "-".join(base.split())
    return base or "untitled"

# ---------- Routes ----------
@docs_bp.route("/", methods=["GET"])
@login_required
def index():
    # simple listing
    docs = DocModel.query.order_by(DocModel.id.desc()).limit(200).all()
    return render_template("docs_index.html", docs=docs)

@docs_bp.route("/new", methods=["GET"])
@login_required
def new_form():
    return render_template("docs_new.html")

@docs_bp.route("/save", methods=["POST"])
@login_required
def save_document():
    """
    Create or update a document.
    Accepts either form-encoded or JSON:
      - id (optional) -> update if provided
      - title
      - content
      - format: 'markdown' or 'latex'
      - visibility (optional), tags (optional)
    Returns JSON with persisted fields and artifact paths.
    """
    data = request.get_json(silent=True) or request.form

    doc_id = data.get("id")
    title = (data.get("title") or "").strip()
    content = data.get("content") or ""
    fmt = (data.get("format") or "markdown").lower()
    visibility = data.get("visibility")
    tags = data.get("tags")

    if fmt not in ("markdown", "latex"):
        return jsonify({"error": "format must be 'markdown' or 'latex'"}), 400

    # Create or fetch
    if doc_id:
        doc = DocModel.query.get(int(doc_id))
        if not doc:
            return jsonify({"error": f"document {doc_id} not found"}), 404
    else:
        doc = DocModel()  # avoid kwargs to dodge __init__ signature issues
        db.session.add(doc)

    # Assign fields defensively (only if they exist on the model)
    if hasattr(doc, "title"):
        doc.title = title or getattr(doc, "title", None) or "Untitled"
    if hasattr(doc, "slug"):
        doc.slug = slugify(title) if title else getattr(doc, "slug", None) or f"doc-{datetime.utcnow().timestamp():.0f}"
    if hasattr(doc, "content"):
        doc.content = content
    if hasattr(doc, "format"):
        doc.format = fmt
    if hasattr(doc, "visibility") and visibility is not None:
        doc.visibility = visibility
    if hasattr(doc, "tags") and tags is not None:
        doc.tags = tags
    if hasattr(doc, "user_id") and current_user and getattr(current_user, "id", None):
        doc.user_id = current_user.id
    if hasattr(doc, "updated_at"):
        doc.updated_at = datetime.utcnow()
    if hasattr(doc, "created_at") and getattr(doc, "created_at", None) is None:
        doc.created_at = datetime.utcnow()

    # Ensure per-doc directory exists
    db.session.flush()  # get doc.id for new docs
    ddir = _doc_dir(doc.id)

    # Persist source file
    if fmt == "markdown":
        src_path = os.path.join(ddir, "document.md")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Render HTML snapshot (sanitized)
        html = render_markdown_safe(content)
        html_path = os.path.join(ddir, "document.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        if hasattr(doc, "source_path"): doc.source_path = src_path
        if hasattr(doc, "rendered_html_path"): doc.rendered_html_path = html_path
        if hasattr(doc, "compiled_pdf_path"): doc.compiled_pdf_path = None

    else:  # latex
        tex_path = os.path.join(ddir, "document.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(content)

        pdf_path = compile_latex_to_pdf(tex_path, ddir)
        if hasattr(doc, "source_path"): doc.source_path = tex_path
        if hasattr(doc, "rendered_html_path"): doc.rendered_html_path = None
        if hasattr(doc, "compiled_pdf_path"): doc.compiled_pdf_path = pdf_path

    db.session.commit()

    return jsonify({
        "id": doc.id,
        "slug": getattr(doc, "slug", None),
        "title": getattr(doc, "title", None),
        "format": fmt,
        "source_path": getattr(doc, "source_path", None),
        "html_path": getattr(doc, "rendered_html_path", None),
        "pdf_path": getattr(doc, "compiled_pdf_path", None),
        "message": "saved"
    }), 200

@docs_bp.route("/<int:doc_id>", methods=["GET"])
@login_required
def view_document(doc_id: int):
    doc = DocModel.query.get_or_404(doc_id)
    # Prefer HTML if markdown; otherwise embed PDF if present.
    html_path = getattr(doc, "rendered_html_path", None)
    pdf_path = getattr(doc, "compiled_pdf_path", None)
    return render_template("docs_view.html", doc=doc, html_path=html_path, pdf_path=pdf_path)

@docs_bp.route("/<int:doc_id>/download/<string:kind>", methods=["GET"])
@login_required
def download_artifact(doc_id: int, kind: str):
    doc = DocModel.query.get_or_404(doc_id)
    kind = kind.lower()
    if kind == "source":
        p = getattr(doc, "source_path", None)
    elif kind == "html":
        p = getattr(doc, "rendered_html_path", None)
    elif kind == "pdf":
        p = getattr(doc, "compiled_pdf_path", None)
    else:
        abort(400, "kind must be one of: source, html, pdf")

    if not p or not os.path.exists(p):
        abort(404, f"{kind} not available for this document")

    fname = os.path.basename(p)
    return send_file(p, as_attachment=True, download_name=fname)
