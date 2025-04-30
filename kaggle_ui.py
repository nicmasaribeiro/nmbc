from flask import Blueprint, render_template, request, redirect, url_for, send_file,send_from_directory,jsonify
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
import json

kaggle_bp = Blueprint("app", __name__)

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
            return redirect(url_for("app.my_notebooks"))

        return "Invalid file format. Please upload a .ipynb file.", 400

    return render_template("submit_notebook.html")


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
def evaluate():
    try:
        data = request.get_json(force=True)
        code = data.get("code", "")

        buffer = io.StringIO()
        local_vars = {}

        with contextlib.redirect_stdout(buffer):
            exec(code, {}, local_vars)  # Run code and capture stdout

        output = buffer.getvalue()
        return jsonify({ "result": output or "✅ No output" })

    except Exception as e:
        return jsonify({ "result": f"❌ Syntax Error: {str(e)}" })


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
        download_link=url_for('app.download_notebook')
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
    return redirect(url_for('app.notebook_manager'))

@kaggle_bp.route("/datasets/upload", methods=["GET", "POST"])
@login_required
def upload_dataset():
    if request.method == "POST":
        file = request.files["dataset"]
        description = request.form.get("description", "")
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join("datasets", filename)
            file.save(save_path)
            from models import DatasetMeta
            # Optional: save metadata to the database
            new_entry = DatasetMeta(
                filename=filename,
                description=description,
                user_id=current_user.id
            )
            db.session.add(new_entry)
            db.session.commit()

            return redirect(url_for("app.list_datasets"))
    return render_template("upload_dataset.html")

from flask import request, jsonify
from nbformat.v4 import new_notebook, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

@kaggle_bp.route("/evaluate_sequence", methods=["POST"])
def evaluate_sequence():
    try:
        cells = request.json.get("cells", [])
        global_env = {}
        results = []

        for cell in cells:
            code = cell["content"]
            try:
                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                    exec(code, global_env)
                    results.append(buf.getvalue())
            except Exception as e:
                results.append(f"❌ {e}")

        return jsonify({ "results": results })

    except Exception as err:
        return jsonify({ "error": str(err) }), 500

@kaggle_bp.route("/sequential/save", methods=["POST"])
def save_sequential_notebook():
    try:
        data = request.get_json()
        name = data.get("name", "Untitled Sequential Notebook")
        notebook = data.get("notebook", [])

        if not isinstance(notebook, list):
            return jsonify({"error": "Notebook must be a list of cells"}), 400

        # Save to DB
        new_entry = UserNotebook(
            user_id=current_user.id,
            name=name,
            content=json.dumps(notebook),
            is_sequential=True,
        )
        db.session.add(new_entry)
        db.session.commit()

        return jsonify({"message": "✅ Notebook saved!"})
    except Exception as e:
        return jsonify({"error": f"❌ Save error: {str(e)}"}), 500

@kaggle_bp.route("/sequential/download/<int:notebook_id>")
@login_required
def download_sequential_notebook(notebook_id):
    from models import UserNotebook
    import io
    import json

    nb = UserNotebook.query.get_or_404(notebook_id)

    if nb.user_id != current_user.id:
        return "Unauthorized", 403

    try:
        notebook_cells = json.loads(nb.content)

        # Build standard .ipynb structure
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

        for cell in notebook_cells:
            lines = cell.get("content", "").splitlines(keepends=True)
            notebook["cells"].append({
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": lines
            })

        # Create in-memory download
        file_buffer = io.BytesIO()
        file_buffer.write(json.dumps(notebook, indent=2).encode("utf-8"))
        file_buffer.seek(0)

        return send_file(
            file_buffer,
            as_attachment=True,
            download_name=f"{nb.name.replace(' ', '_')}.ipynb",
            mimetype="application/json"
        )
    except Exception as e:
        return f"❌ Download failed: {e}", 500

@kaggle_bp.route("/sequential/load/<int:notebook_id>")
@login_required
def load_sequential_notebook(notebook_id):
    nb = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id, is_sequential=True).first()
    if not nb:
        return "Notebook not found", 404

    return jsonify({
        "name": nb.name,
        "cells": json.loads(nb.content)
    })


@kaggle_bp.route("/sequential")
@login_required
def sequential_notebook():
    notebook = UserNotebook.query.filter_by(user_id=current_user.id, is_sequential=True).order_by(UserNotebook.updated_at.desc()).first()
    return render_template("sequential_notebook.html", notebook=notebook)




@kaggle_bp.route("/datasets")
@login_required
def list_datasets():
    from models import DatasetMeta
    files = DatasetMeta.query.order_by(DatasetMeta.uploaded_at.desc()).all()
    return render_template("list_datasets.html", files=files)



@kaggle_bp.route("/save_user_notebook", methods=["POST"])
@login_required
def save_user_notebook():
    try:
        data = request.get_json()
        name = data.get("name", "Untitled Notebook")
        notebook_data = data.get("notebook", [])
        if not notebook_data:
            return "❌ Empty notebook", 400

        new_entry = UserNotebook(
            user_id=current_user.id,
            name=name,
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


@kaggle_bp.route("/runner")
@login_required
def runner():
    return render_template("runner.html")

@kaggle_bp.route("/notebook/rename/<int:notebook_id>", methods=["POST"])
@login_required
def rename_notebook(notebook_id):
    new_name = request.form.get("new_name", "").strip()
    if not new_name:
        return "⚠️ Name cannot be empty.", 400

    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not notebook:
        return "Notebook not found", 404

    notebook.name = new_name
    db.session.commit()
    return redirect(url_for("app.my_notebooks"))


@kaggle_bp.route("/edit_notebook/<int:notebook_id>")
@login_required
def edit_saved_notebook(notebook_id):
    nb = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not nb:
        return "Notebook not found", 404

    try:
        # Parse to ensure it's valid JSON, then re-dump it for safe JS usage
        parsed = json.loads(nb.content)
        encoded = json.dumps(parsed)
    except Exception as e:
        print("⚠️ JSON parsing error:", e)
        encoded = "[]"
    import json

    
    notebook = UserNotebook.query.get(notebook_id)
    parsed = json.loads(notebook.content)       # parsed: list of dicts
    safe_json = json.dumps(parsed)              # string with real \n, not \\n

    return render_template("editor.html", saved_cells=safe_json, notebook_id=nb.id, notebook_name=nb.name)

    # return render_template(
    #     "editor.html",
    #     notebook_name=nb.name,
    #     saved_cells=encoded
    # )



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



@kaggle_bp.route("/run", methods=["GET", "POST"])
def run():
    import requests as rq

    if request.method == "POST":
        file = request.files.get("files")
        if not file:
            return "❌ No file uploaded", 400

        code = file.read().decode("utf-8")

        try: #http://127.0.0.1:9090/execute
            response = rq.post("https://nmbc-executer.onrender.com/execute", json={"code": code})
            result = response.json().get("result", "No output.")
            return render_template("execute.html", output=result)
        except Exception as e:
            return f"❌ Execution failed: {str(e)}", 500

    return render_template("execute.html", output=None)


@kaggle_bp.route("/datasets/download/<filename>")
@login_required
def download_dataset(filename):
    import os
    from werkzeug.utils import secure_filename

    safe_filename = secure_filename(filename)
    path = os.path.join("datasets", safe_filename)

    if not os.path.exists(path):
        return abort(404)

    return send_from_directory("datasets", safe_filename, as_attachment=True)

@kaggle_bp.route("/create_ipynb", methods=["POST"])
def create_ipynb():
    try:
        data = request.get_json()
        notebook_cells = data.get("notebook", [])

        # Build notebook structure
        nb = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

        for cell in notebook_cells:
            content = cell.get("content", "")
            cell_type = cell.get("type", "code")
            lines = content.splitlines(keepends=False)  # this avoids inline \n

            if cell_type == "code":
                nb["cells"].append({
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": lines
                })
            elif cell_type == "markdown":
                nb["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": lines
                })

        # Save to a file (optional: unique name)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"notebook_{timestamp}.ipynb"
        save_path = os.path.join("submissions", filename)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)

        return jsonify({"message": "Notebook saved", "filename": filename}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@kaggle_bp.route("/notebook/<int:notebook_id>/tags", methods=["POST"])
@login_required
def update_notebook_tags(notebook_id):
    notebook = UserNotebook.query.get_or_404(notebook_id)
    if notebook.user_id != current_user.id:
        abort(403)
    new_tags = request.form.get("tags", "")
    notebook.tags = new_tags
    db.session.commit()
    return redirect(url_for("app.my_notebooks"))

from datetime import datetime
@kaggle_bp.route("/notebook/publish/<int:notebook_id>", methods=["POST"])
@login_required
def publish_notebook(notebook_id):
    nb = UserNotebook.query.get_or_404(notebook_id)

    if nb.user_id != current_user.id:
        abort(403)

    nb.published = True
    nb.published_at = datetime.utcnow()
    db.session.commit()

    return redirect(url_for("app.my_notebooks"))

@kaggle_bp.route("/explore")
def explore_notebooks():
    published = UserNotebook.query.filter_by(published=True).order_by(UserNotebook.published_at.desc()).all()
    return render_template("explore_public.html", notebooks=published)

@kaggle_bp.route("/notebook/toggle/<int:notebook_id>", methods=["POST"])
@login_required
def toggle_publish_status(notebook_id):
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not notebook:
        return "Notebook not found or unauthorized", 404

    notebook.published = not notebook.published
    if notebook.published:
        from datetime import datetime
        notebook.published_at = datetime.utcnow()
    else:
        notebook.published_at = None

    db.session.commit()
    return redirect(url_for("app.my_notebooks"))

@kaggle_bp.route("/notebook/update/<int:notebook_id>", methods=["POST"])
@login_required
def update_notebook(notebook_id):
    try:
        data = request.get_json()
        new_cells = data.get("notebook", [])
        new_name = data.get("name")

        notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
        if not notebook:
            return "Notebook not found or unauthorized", 404

        if new_name:
            notebook.name = new_name
        if new_cells:
            notebook.content = json.dumps(new_cells)

        db.session.commit()
        return "✅ Notebook updated", 200

    except Exception as e:
        print("❌ Update error:", e)
        return f"❌ Failed to update notebook: {str(e)}", 500
    
    from flask import send_from_directory
from nbconvert import HTMLExporter
import nbformat

@kaggle_bp.route("/notebook/view/<filename>")
def view_notebook(filename):
    import os

    path = os.path.join("submissions", filename)
    if not os.path.exists(path):
        return f"Notebook {filename} not found", 404

    # Convert notebook to HTML using nbconvert
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    html_exporter.exclude_input_prompt = True
    html_exporter.exclude_output_prompt = True
    body, _ = html_exporter.from_notebook_node(nb)

    # Save the HTML output
    out_file = os.path.join("rendered_notebooks", f"{filename}.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(body)

    return redirect(url_for('app.notebook_snapshot', htmlfile=f"{filename}.html"))
from flask import render_template, send_from_directory

@kaggle_bp.route("/snapshot/<filename>")
def notebook_snapshot(filename):
    # Make sure the file exists in rendered_notebooks/
    path = os.path.join("rendered_notebooks", filename)
    if not os.path.exists(path):
        return "Notebook not found", 404
    return render_template("notebook_snapshot.html", htmlfile=filename)

@kaggle_bp.route("/notebook/set_tags/<int:notebook_id>", methods=["POST"])
@login_required
def set_notebook_tags(notebook_id):
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not notebook:
        return "Notebook not found", 404
    tags = request.form.get("tags", "").strip()
    if tags:
        notebook.tags = tags  # assumes you added a `tags` column
        db.session.commit()
    return redirect(url_for("app.my_notebooks"))


@kaggle_bp.route("/notebook/snapshot")
def notebook_snapshot_two():
    htmlfile = request.args.get("htmlfile")
    return render_template("notebook_snapshot.html", htmlfile=htmlfile)

@kaggle_bp.route("/editor", methods=["GET"])
def monaco_editor():
    return render_template(
        "editor.html",
        saved_cells=[],
        notebook_name="Untitled Notebook",
        notebook_id="null"
    )

@kaggle_bp.route("/edit_notebook/<int:notebook_id>")
@login_required
def edit_notebook(notebook_id):
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first_or_404()
    try:
        cells = json.loads(notebook.content)
    except Exception as e:
        cells = []

    return render_template(
        "editor.html",
        saved_cells=cells,
        notebook_name=notebook.name,
        notebook_id=notebook.id
    )


@kaggle_bp.route("/jup")
def jup():
    return render_template("jup_lite.html")

@kaggle_bp.route("/rendered/<filename>")
def rendered_file(filename):
    return send_from_directory("rendered_notebooks", filename)

@kaggle_bp.route("/sequential/open/<int:notebook_id>")
@login_required
def open_sequential_notebook_two(notebook_id):
    nb = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id, is_sequential=True).first()
    if not nb:
        return "Notebook not found", 404

    return render_template("sequential_notebook.html", notebook=nb)

@kaggle_bp.route("/sequential/<int:notebook_id>")
@login_required
def open_sequential_notebook(notebook_id):
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not notebook:
        return "Notebook not found or unauthorized", 404

    # You can customize the UI to look different for sequential view if needed
    return render_template("sequential_view.html", notebook=notebook)
