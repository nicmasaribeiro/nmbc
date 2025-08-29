# docs_blueprint.py
from __future__ import annotations
import os, shutil, tempfile, subprocess, shlex
from pathlib import Path
from flask import Blueprint, request, render_template, redirect, url_for, flash, abort, send_file, jsonify, current_app
from flask_login import login_required, current_user
from markdown import markdown
import bleach

from models import db, Notebook  # using your existing model

docs = Blueprint("docs_bp", __name__, template_folder="templates")

# ---- Markdown sanitize ----
ALLOWED_TAGS = bleach.sanitizer.ALLOWED_TAGS.union({
    "p","pre","code","h1","h2","h3","h4","h5","h6","span","div","hr",
    "img","table","thead","tbody","tr","th","td","blockquote","ul","ol","li"
})
ALLOWED_ATTRS = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "img": ["src","alt","title"],
    "a": ["href","title","name","target","rel"],
    "span": ["class"],
    "div": ["class","id"]
}
def render_markdown_safe(text: str) -> str:
    raw = markdown(text, extensions=["extra", "codehilite", "tables", "toc"])
    clean = bleach.clean(raw, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
    return bleach.linkify(clean)

def has_pdflatex() -> bool:
    return shutil.which("pdflatex") is not None

# ---------- LIST ----------
@docs.route("/docs/list")
@login_required
def list_docs():
    items = Notebook.query.filter_by(user=current_user.username).order_by(Notebook.id.desc()).all()
    return render_template("list_docs.html", docs=items)

# ---------- PREVIEW ----------
@docs.route("/docs/preview/<int:doc_id>")
@login_required
def preview_doc(doc_id):
    doc = Notebook.query.filter_by(id=doc_id, user=current_user.username).first_or_404()
    if (doc.file_type or "").lower() in ("md", "markdown"):
        html = render_markdown_safe(doc.content or "")
        return render_template("preview_markdown.html", doc=doc, html=html)

    # default: latex preview (no server pdflatex required)
    # We pass raw LaTeX to the template; latex.js/MathJax render it in the browser.
    return render_template("preview_latex.html", doc=doc, latex_source=doc.content or "", pdflatex_available=has_pdflatex())

# ---------- (Optional) PDF compile if pdflatex exists ----------
def _compile_latex_to_pdf(tex_source: str, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    main_tex = Path(out_dir) / "main.tex"

    s_low = (tex_source or "").lower()
    needs_wrapper = ("\\documentclass" not in s_low) or ("\\begin{document}" not in s_low)
    if needs_wrapper:
        tex_source = (
            r"\documentclass[11pt]{article}"
            r"\usepackage[utf8]{inputenc}\usepackage{amsmath,amssymb,amsthm}"
            r"\usepackage{geometry}\usepackage{hyperref}\geometry{margin=1in}"
            r"\begin{document}"
            + (tex_source or "")
            + r"\end{document}"
        )

    main_tex.write_text(tex_source, encoding="utf-8")
    cmd = "pdflatex -interaction=nonstopmode -halt-on-error main.tex"
    for _ in range(2):
        proc = subprocess.run(shlex.split(cmd), cwd=out_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            out = proc.stdout.decode("utf-8", errors="ignore")
            raise RuntimeError("LaTeX compile error:\n" + out[:8000])
    return str(Path(out_dir) / "main.pdf")

@docs.route("/docs/pdf/<int:doc_id>")
@login_required
def pdf_doc(doc_id):
    if not has_pdflatex():
        return "pdflatex is not installed on the server.", 422
    doc = Notebook.query.filter_by(id=doc_id, user=current_user.username).first_or_404()
    tmp = tempfile.mkdtemp()
    try:
        pdf_path = _compile_latex_to_pdf(doc.content or "", tmp)
        return send_file(pdf_path, as_attachment=False, download_name=f"{(doc.title or 'document')}.pdf")
    except Exception as e:
        return f"❌ Compile error: {e}", 500
