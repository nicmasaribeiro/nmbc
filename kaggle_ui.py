from flask import Blueprint, render_template, request, redirect, url_for, send_file,send_from_directory
from werkzeug.utils import secure_filename
from models import NotebookSubmission, db
import nbformat
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from flask_login import login_required, current_user
from nbformat.v4 import new_notebook, new_code_cell
import io
import contextlib
import requests
import time
from models import UserNotebook
from flask import abort

kaggle_bp = Blueprint("kaggle", __name__)

UPLOAD_FOLDER = "./submissions"
ALLOWED_EXTENSIONS = {"ipynb"}
NOTEBOOK_FOLDER = "./submissions"

# At the top of kaggle_ui.py
def execute_code_cells(code_cells):
    nb = new_notebook(cells=[new_code_cell(source=code) for code in code_cells])
    ep = ExecutePreprocessor(timeout=60, kernel_name="python3")
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        return nb, f"Execution error: {e}"
    return nb, None


@kaggle_bp.route("/notes")
@login_required
def notebook_manager():
    notebooks = UserNotebook.query.filter_by(user_id=current_user.id).order_by(UserNotebook.updated_at.desc()).all()
    return render_template("notebook_manager.html", notebooks=notebooks)


from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_notebook, new_code_cell
import json

def execute_notebook_and_capture(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=60, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        print("⚠️ Execution error:", e)

    # Save back the executed notebook
    with open(path, "w") as f:
        nbformat.write(nb, f)

    return nb



@kaggle_bp.route("/")
def kaggle_home():
    submissions = NotebookSubmission.query.order_by(NotebookSubmission.score.desc()).all()
    return render_template("kaggle_home.html", submissions=submissions)


@kaggle_bp.route("/submit", methods=["GET", "POST"])
@login_required
def submit_notebook():
    if request.method == "POST":
        file = request.files["notebook"]
        if file and file.filename.endswith(".ipynb"):
            filename = secure_filename(file.filename)
            full_path = os.path.join("submissions", f"{current_user.id}_{filename}")
            file.save(full_path)

            # ✅ Execute notebook and get outputs
            nb = execute_notebook_and_capture(full_path)

            # ✅ Optionally extract score from last cell
            score = 0.0
            for cell in reversed(nb.cells):
                if cell.cell_type == "code" and cell.outputs:
                    for output in cell.outputs:
                        if output.output_type == "execute_result":
                            try:
                                score = float(output.data["text/plain"])
                            except Exception:
                                pass
                            break

            # ✅ Create leaderboard entry
            submission = NotebookSubmission(
                user_id=current_user.id,
                notebook_filename=filename,
                score=score
            )
            db.session.add(submission)

            # ✅ Save cell contents and output into DB
            notebook_payload = []
            for cell in nb.cells:
                output = []
                if cell.cell_type == "code":
                    for o in cell.get("outputs", []):
                        if o.output_type == "stream":
                            output.append(o.text)
                        elif o.output_type == "execute_result":
                            output.append(o['data'].get('text/plain', ''))
                        elif o.output_type == "error":
                            output.append('Error: ' + '\n'.join(o['traceback']))

                notebook_payload.append({
                    "type": cell.cell_type,
                    "content": cell.source,
                    "output": output
                })

            db_notebook = UserNotebook(
                user_id=current_user.id,
                name=f"Submitted: {filename}",
                content=json.dumps(notebook_payload)
            )
            db.session.add(db_notebook)

            db.session.commit()
            return redirect(url_for("kaggle.my_notebooks"))

        return "Invalid file format. Please upload a .ipynb file.", 400

    return render_template("submit_notebook.html")


@kaggle_bp.route("/editor", methods=["GET"])
def monaco_editor():
    return render_template("editor.html")

@kaggle_bp.route("/editor/save", methods=["POST"])
def save_notebook_from_editor():
    data = request.get_json()
    name = data.get("name", "Untitled")
    cells = data.get("notebook", [])

    from models import Notebook
    import json

    nb = Notebook(
        name=name,
        content=json.dumps(cells),
        user_id=getattr(current_user, 'id', None)
    )
    db.session.add(nb)
    db.session.commit()

    return "✅ Notebook saved", 200

@kaggle_bp.route("/editor/load/<int:notebook_id>")
def load_notebook(notebook_id):
    from models import Notebook
    nb = Notebook.query.get(notebook_id)
    if not nb:
        return "Notebook not found", 404

    return jsonify({
        "name": nb.name,
        "cells": json.loads(nb.content)
    })

@kaggle_bp.route("/evaluate", methods=["POST"])
def evaluate_code():
    code = request.form.get("code", "").strip()

    if not code:
        return "<pre>⚠️ No code submitted.</pre>"

    try:
        start = time.time()
        nb, error = execute_code_cells([code])
        duration = round(time.time() - start, 3)

        output = []
        for cell in nb.cells:
            for output_obj in cell.get('outputs', []):
                if output_obj.output_type == 'stream':
                    output.append(output_obj.text)
                elif output_obj.output_type == 'execute_result':
                    output.append(output_obj['data'].get('text/plain', ''))
                elif output_obj.output_type == 'error':
                    output.append('Error: ' + '\n'.join(output_obj['traceback']))

        result = '\n'.join(output) if output else "✅ Executed successfully with no output."
        if error:
            result = f"❌ {error}"

        return f"<pre>{result}</pre><p><em>⏱️ {duration}s</em></p>"

    except Exception as e:
        return f"<pre>❌ Exception: {e}</pre>"


@kaggle_bp.route("/open_notebook", methods=["GET", "POST"])
def open_notebook():
    if request.method == "POST":
        file = request.files["notebook"]
        if file.filename.endswith(".ipynb"):
            nb = nbformat.read(file, as_version=4)
            # Extract source code of code cells
            code_cells = [cell["source"] for cell in nb.cells if cell.cell_type == "code"]
            return render_template("notebook_editor.html", cells=code_cells)
    return render_template("upload_notebook_two.html")


@kaggle_bp.route("/save_notebook", methods=["POST"])
@login_required
def save_notebook():
    cells = request.form.getlist("cells")
    nb = nbformat.v4.new_notebook()
    nb.cells = []

    for cell in cells:
        if cell.startswith("#MD#"):
            nb.cells.append(nbformat.v4.new_markdown_cell(source=cell.replace("#MD#", "", 1)))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source=cell))

    ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        pass

    filename = secure_filename(f"{current_user.id}_executed_notebook.ipynb")
    output_path = os.path.join("./submissions", filename)
    with open(output_path, "w") as f:
        nbformat.write(nb, f)

    # ✅ Save to database
    try:
        notebook_payload = [
            {
                "type": cell.cell_type,
                "content": cell.source if isinstance(cell.source, str) else ''.join(cell.source)
            } for cell in nb.cells
        ]
        new_entry = UserNotebook(
            user_id=current_user.id,
            name="Executed Notebook",
            content=json.dumps(notebook_payload)
        )
        db.session.add(new_entry)
        db.session.commit()
    except Exception as e:
        print("❌ Error saving to database:", e)

    # Render output
    rendered_cells = []
    cell_outputs = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            outputs = []
            for output in cell.get('outputs', []):
                if output.output_type == 'stream':
                    outputs.append(output.text)
                elif output.output_type == 'execute_result':
                    outputs.append(output['data'].get('text/plain', ''))
                elif output.output_type == 'error':
                    outputs.append('Error: ' + '\n'.join(output['traceback']))
            cell_outputs.append(outputs)
            rendered_cells.append(cell.source)
        elif cell.cell_type == "markdown":
            cell_outputs.append([])
            rendered_cells.append(cell.source)

    paired_cells = list(zip(rendered_cells, cell_outputs))

    return render_template(
        "execution_result.html",
        paired_cells=paired_cells,
        download_link=url_for('kaggle.download_notebook')
    )



@kaggle_bp.route("/download_notebook")
def download_notebook():
    filename = f"{current_user.id}_executed_notebook.ipynb"
    return send_file(os.path.join("submissions", filename), as_attachment=True)

@kaggle_bp.route("/notebook/download/<filename>")
@login_required
def download_notebook_file(filename):
    # Check that file belongs to the current user
    if not filename.startswith(f"{current_user.id}_"):
        return "Unauthorized access", 403  # Forbidden
    
    return send_from_directory(NOTEBOOK_FOLDER, filename, as_attachment=True)

@kaggle_bp.route("/notebook/delete/<filename>", methods=["POST"])
@login_required
def delete_notebook_file(filename):
    if not filename.startswith(f"{current_user.id}_"):
        return "Unauthorized access", 403  # Forbidden

    file_path = os.path.join(NOTEBOOK_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('kaggle.notebook_manager'))

@kaggle_bp.route("/datasets/upload", methods=["GET", "POST"])
@login_required
def upload_dataset():
    if request.method == "POST":
        file = request.files["dataset"]
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join("datasets", filename)
            file.save(path)
            return redirect(url_for("kaggle.list_datasets"))
    return render_template("upload_dataset.html")

@kaggle_bp.route("/datasets")
@login_required
def list_datasets():
    files = [f for f in os.listdir("datasets") if not f.startswith(".")]
    return render_template("list_datasets.html", files=files)


@kaggle_bp.route("/kaggle/save_user_notebook", methods=["POST"])
@login_required
def save_user_notebook():
    try:
        data = request.get_json()
        print("✅ Received notebook:", data)

        notebook_data = data.get("notebook", [])
        if not notebook_data:
            return "❌ Empty notebook", 400

        from models import UserNotebook
        import json

        new_entry = UserNotebook(
            user_id=current_user.id,
            name="Untitled Notebook",
            content=json.dumps(notebook_data)
        )
        db.session.add(new_entry)
        db.session.commit()
        return "✅ Notebook saved!", 200

    except Exception as e:
        print("❌ Save error:", e)
        return f"❌ Save failed: {e}", 500


@kaggle_bp.route("/my_notebooks")
@login_required
def my_notebooks():
    notebooks = UserNotebook.query.filter_by(user_id=current_user.id).order_by(UserNotebook.updated_at.desc()).all()
    return render_template("notebooks_two.html", notebooks=notebooks)

@kaggle_bp.route("/kaggle/edit_notebook/<int:notebook_id>")
@login_required
def edit_saved_notebook(notebook_id):
    from models import UserNotebook
    import json

    nb = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not nb:
        return "Notebook not found or unauthorized", 404

    # Pass notebook name and parsed cells to editor
    return render_template("editor.html", notebook_name=nb.name, saved_cells=nb.content)


@kaggle_bp.route("/notebook/open/<filename>")
@login_required
def load_uploaded_notebook(filename):
    
    # Check if the file belongs to the current user
    if not filename.startswith(f"{current_user.id}_"):
        return "Unauthorized", 403

    file_path = os.path.join(NOTEBOOK_FOLDER, filename)

    if not os.path.exists(file_path):
        abort(404)

    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Extract source code from code cells
    code_cells = [cell["source"] for cell in nb.cells if cell.cell_type == "code"]

    return render_template("notebook_editor.html", cells=code_cells)

