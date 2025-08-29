# docs_blueprint.py
from flask import Blueprint, render_template
from models import Document

docs_bp = Blueprint("docs", __name__, template_folder="templates")

@docs_bp.route("/view/<slug>")
def view_document(slug):
    doc = Document.query.filter_by(slug=slug).first_or_404()
    return render_template("docs/view.html", doc=doc)

@docs_bp.route("/list")
def list_documents():
    docs = Document.query.order_by(Document.created_at.desc()).all()
    return render_template("docs/list.html", docs=docs)
