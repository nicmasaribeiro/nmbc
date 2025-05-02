from flask import Blueprint, request, render_template, jsonify
from flask_login import login_required, current_user
from models import db, UserNotebook
import json
from datetime import datetime

sequential_bp = Blueprint("sequential", __name__, template_folder="templates")

@sequential_bp.route("/editor")
@login_required
def editor():
    return render_template("code.html")

@sequential_bp.route("/app/sequential/save", methods=["POST"])
@login_required
def save_notebook():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        name = data.get("name")
        if not name:
            return jsonify({"error": "Notebook name is required"}), 400
            
        notebook_content = data.get("notebook")
        if not isinstance(notebook_content, list):
            return jsonify({"error": "Invalid notebook format"}), 400

        # Check if notebook exists for this user with the same name
        existing_notebook = UserNotebook.query.filter_by(
            user_id=current_user.id,
            name=name
        ).first()

        if existing_notebook:
            # Update existing notebook
            existing_notebook.content = json.dumps(notebook_content)
            existing_notebook.updated_at = datetime.utcnow()
            db.session.commit()
            return jsonify({
                "message": "Notebook updated successfully!",
                "id": existing_notebook.id
            })
        else:
            # Create new notebook
            notebook = UserNotebook(
                user_id=current_user.id,
                name=name,
                content=json.dumps(notebook_content)
            )
            db.session.add(notebook)
            db.session.commit()
            return jsonify({
                "message": "Notebook saved successfully!",
                "id": notebook.id
            })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to save notebook: {str(e)}"}), 500

@sequential_bp.route("/app/sequential/list")
@login_required
def list_notebooks():
    try:
        notebooks = UserNotebook.query.filter_by(
            user_id=current_user.id
        ).order_by(UserNotebook.updated_at.desc()).all()
        
        return jsonify([
            {
                "id": n.id,
                "name": n.name,
                "updated_at": n.updated_at.isoformat()
            } for n in notebooks
        ])
    except Exception as e:
        return jsonify({"error": f"Failed to list notebooks: {str(e)}"}), 500

@sequential_bp.route("/app/sequential/load")
@login_required
def load_notebook():
    try:
        notebook_id = request.args.get("id")
        if not notebook_id:
            return jsonify({"error": "Notebook ID is required"}), 400

        notebook = UserNotebook.query.filter_by(
            id=notebook_id,
            user_id=current_user.id
        ).first()

        if not notebook:
            return jsonify({"error": "Notebook not found"}), 404

        return jsonify({
            "name": notebook.name,
            "cells": json.loads(notebook.content),
            "id": notebook.id
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load notebook: {str(e)}"}), 500