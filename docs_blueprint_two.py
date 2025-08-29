# docs_blueprint.py
import os, shutil, subprocess, json
from pathlib import Path
from datetime import datetime

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    send_file, abort, current_app, jsonify
)
from markupsafe import Markup
import markdown as md

from models import db, Document

docs_bp_two = Blueprint("docs", __name__, template_folder="templates")

# ---------- Helpers ----------
def _storage_root():
    root = os.path.join(current_app.instance_path, "docs")
    Path(root).mkdir(parents=True, exist_ok=True)
    return root

def render_markdown_safe(text: str) -> str:
    return md.markdown(text or "", extensions=["fenced_code", "tables", "toc", "codehilite"])

def compile_latex_to_pdf(tex_source: str, out_dir: str) -> str:
    """Compile LaTeX to PDF using pdflatex (if available). Raises helpful error if missing."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tex_file = out / "document.tex"
    tex_file.write_text(tex_source, encoding="utf-8")

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise FileNotFoundError("pdflatex not found in PATH. Install TeX Live/MacTeX or set PATH.")

    # Run twice for refs/TOC consistency
    for _ in range(2):
        proc = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if proc.returncode != 0:
            # Keep log to help debugging
            (out / "compile.log").write_text(proc.stdout or "", encoding="utf-8")
            raise RuntimeError("LaTeX compile failed. See compile.log in the doc folder.")

    pdf_path = out / "document.pdf"
    return str(pdf_path)

def _ensure_rendered(doc: Document):
    doc_dir = os.path.join(_storage_root(), str(doc.id))
    Path(doc_dir).mkdir(parents=True, exist_ok=True)

    if doc.content_type == "markdown":
        html = render_markdown_safe(doc.content)
        rendered_path = os.path.join(doc_dir, "document.html")
        Path(rendered_path).write_text(html, encoding="utf-8")
        doc.rendered_html_path = rendered_path
        db.session.commit()
    else:
        # Only (re)compile if no compiled PDF yet
        if not doc.compiled_pdf_path or not Path(doc.compiled_pdf_path).exists():
            pdf_path = compile_latex_to_pdf(doc.content, doc_dir)
            doc.compiled_pdf_path = pdf_path
            db.session.commit()

# ---------- Routes ----------
@docs_bp_two.route("/docs/")
def list_documents():
    docs = Document.query.order_by(Document.created_at.desc()).all()
    return render_template("docs/list.html", docs=docs)

@docs_bp_two.route("/docs/new", methods=["GET", "POST"])
def new_document():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content_type = request.form.get("content_type", "markdown").strip().lower()
        content = request.form.get("content", "")
        owner_id = getattr(getattr(request, "user", None), "id", None)  # adapt if needed

        if content_type not in ("markdown", "latex"):
            flash("Invalid content type.", "error")
            return render_template("docs/new.html", default_type="markdown"), 400
        if not title or not content:
            flash("Title and content are required.", "error")
            return render_template("docs/new.html",
                                   default_type=content_type, title=title, content=content), 400

        doc = Document(title=title, content_type=content_type, content=content, owner_id=owner_id)
        db.session.add(doc)
        db.session.commit()

        doc.ensure_slug()
        db.session.commit()

        # Persist initial artifacts (best effort)
        doc_dir = os.path.join(_storage_root(), str(doc.id))
        Path(doc_dir).mkdir(parents=True, exist_ok=True)
        try:
            if content_type == "markdown":
                html = render_markdown_safe(content)
                rendered_path = os.path.join(doc_dir, "document.html")
                Path(rendered_path).write_text(html, encoding="utf-8")
                doc.rendered_html_path = rendered_path
            else:
                pdf_path = compile_latex_to_pdf(content, doc_dir)
                doc.compiled_pdf_path = pdf_path
        except Exception as e:
            flash(f"Saved, but initial rendering failed: {e}", "warning")
        finally:
            db.session.commit()

        flash("Document created.", "success")
        return redirect(url_for("docs.view_document", slug=doc.slug))

    return render_template("docs/new.html", default_type="markdown")

@docs_bp_two.route("/docs/<slug>")
def view_document(slug):
    doc = Document.query.filter_by(slug=slug).first()
    if not doc:
        abort(404)

    # Render artifacts if missing (best effort)
    try:
        _ensure_rendered(doc)
    except Exception as e:
        flash(f"Preview not available: {e}", "warning")

    return render_template("docs/view.html", doc=doc)

@docs_bp_two.route("/docs/<slug>/preview")
def preview_document(slug):
    """Inline preview: HTML for Markdown; PDF embedded for LaTeX (if compiled)."""
    doc = Document.query.filter_by(slug=slug).first_or_404()
    try:
        _ensure_rendered(doc)
    except Exception as e:
        flash(f"Preview not available: {e}", "warning")
    return render_template("docs/preview.html", doc=doc)

@docs_bp_two.route("/docs/<slug>/download")
def download_document(slug):
    doc = Document.query.filter_by(slug=slug).first_or_404()
    if doc.content_type == "markdown":
        # Ensure rendered HTML exists
        if not doc.rendered_html_path or not Path(doc.rendered_html_path).exists():
            _ensure_rendered(doc)
        return send_file(doc.rendered_html_path, as_attachment=True, download_name=f"{doc.slug}.html")
    else:
        # LaTeX → PDF
        if not doc.compiled_pdf_path or not Path(doc.compiled_pdf_path).exists():
            _ensure_rendered(doc)
        return send_file(doc.compiled_pdf_path, as_attachment=True, download_name=f"{doc.slug}.pdf")

@docs_bp_two.route("/docs/<slug>/pdf")
def pdf_inline(slug):
    """Serve PDF inline for embedding (LaTeX only)."""
    doc = Document.query.filter_by(slug=slug).first_or_404()
    if doc.content_type != "latex":
        abort(404)
    if not doc.compiled_pdf_path or not Path(doc.compiled_pdf_path).exists():
        _ensure_rendered(doc)
    # Inline
    return send_file(doc.compiled_pdf_path, as_attachment=False)

@docs_bp_two.route("/docs/<slug>/raw")
def raw_document(slug):
    doc = Document.query.filter_by(slug=slug).first_or_404()
    return current_app.response_class(
        response=doc.content or "",
        status=200,
        mimetype="text/plain; charset=utf-8"
    )

@docs_bp_two.route("/docs/<slug>/compile", methods=["POST"])
def recompile_document(slug):
    doc = Document.query.filter_by(slug=slug).first_or_404()
    doc_dir = os.path.join(_storage_root(), str(doc.id))
    Path(doc_dir).mkdir(parents=True, exist_ok=True)

    try:
        if doc.content_type == "markdown":
            html = render_markdown_safe(doc.content)
            rendered_path = os.path.join(doc_dir, "document.html")
            Path(rendered_path).write_text(html, encoding="utf-8")
            doc.rendered_html_path = rendered_path
        else:
            pdf_path = compile_latex_to_pdf(doc.content, doc_dir)
            doc.compiled_pdf_path = pdf_path
        db.session.commit()
        flash("Re-rendered successfully.", "success")
    except Exception as e:
        flash(f"Render error: {e}", "error")
    return redirect(url_for("docs.view_document", slug=doc.slug))

@docs_bp_two.route("/docs/<slug>/edit", methods=["GET", "POST"])
def edit_document(slug):
    doc = Document.query.filter_by(slug=slug).first_or_404()
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "")

        if not title or not content:
            flash("Title and content are required.", "error")
            return render_template("docs/edit.html", doc=doc), 400

        doc.title = title
        doc.content = content
        doc.updated_at = datetime.utcnow()
        db.session.commit()
        flash("Document updated.", "success")
        return redirect(url_for("docs.view_document", slug=doc.slug))

    return render_template("docs/edit.html", doc=doc)

@docs_bp_two.route("/docs/<slug>/delete", methods=["POST"])
def delete_document(slug):
    doc = Document.query.filter_by(slug=slug).first_or_404()
    # remove artifacts folder too
    try:
        doc_dir = os.path.join(_storage_root(), str(doc.id))
        if Path(doc_dir).exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
    finally:
        db.session.delete(doc)
        db.session.commit()
        flash("Document deleted.", "success")
        return redirect(url_for("docs.list_documents"))
