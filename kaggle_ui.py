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
import markdown
# from flask import Markup
import json
from markupsafe import Markup  # Instead of from flask import Markup
import markdown
from nbformat import reads as nbformat_reads
from nbformat import NO_CONVERT
from nbconvert.preprocessors import ExecutePreprocessor
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from flask import current_app
from markupsafe import Markup
import markdown


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

def is_plotting_notebook(notebook_content):
    """Check if notebook contains plotting code"""
    try:
        if isinstance(notebook_content, str):
            content = json.loads(notebook_content)
        else:
            content = notebook_content
            
        plotting_keywords = ['plt.', 'plot(', 'figure(', 'show(', 'matplotlib']
        
        for cell in content:
            if cell.get('type') == 'code':
                code = cell.get('content', '')
                if any(keyword in code for keyword in plotting_keywords):
                    return True
        return False
    except:
        return False

def markdown_to_html(text):
    """Convert markdown text to HTML"""
    if not text:
        return ""
    return Markup(markdown.markdown(text))

def register_template_filters(app):
    """Register custom template filters"""
    app.jinja_env.filters['markdown'] = markdown_to_html

@kaggle_bp.after_request
def after_request(response):
    """Ensure all API responses are JSON"""
    if request.path.startswith('/notebook/'):
        if response.status_code >= 400:
            data = {
                "success": False,
                "message": response.get_data(as_text=True)
            }
            response.set_data(json.dumps(data))
            response.content_type = 'application/json'
    return response

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


@kaggle_bp.route("/index")
def kaggle_home():
    # submissions = NotebookSubmission.query.order_by(NotebookSubmission.score.desc()).all()
    return render_template("kaggle_index.html")#, submissions=submissions)


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

@kaggle_bp.route("/evaluate_two", methods=["POST"])
def evaluate_two():
    data = request.get_json(force=True)
    code = data.get("code", "")

    import io, contextlib
    buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {})
        return jsonify({ "result": buffer.getvalue() })
    except Exception as e:
        return jsonify({ "result": f"❌ {str(e)}" })


@kaggle_bp.route("/open_notebook", methods=["GET", "POST"])
def open_notebook():
    if request.method == "POST":
        file = request.files.get("notebook")
        if not file:
            return "No file uploaded", 400
            
        if not file.filename.endswith(".ipynb"):
            return "Only .ipynb files are supported", 400

        try:
            nb_content = file.read().decode('utf-8')

            try:
                nb = nbformat.reads(nb_content, as_version=4)
            except Exception as e:
                return f"Failed to parse notebook: {str(e)}", 400

            if not hasattr(nb, 'cells') or not isinstance(nb.cells, list):
                return "Invalid notebook format: missing cells", 400

            processed_cells = []
            for cell in nb.cells:
                if not hasattr(cell, 'cell_type') or not hasattr(cell, 'source'):
                    continue
                
                source_lines = cell.source.splitlines(keepends=True) if isinstance(cell.source, str) else cell.source

                cell_data = {
                    "type": cell.cell_type,
                    "content": ''.join(source_lines),  # Keep as string but joined correctly
                    "output": []
                }

                if cell.cell_type == "code":
                    for output in getattr(cell, 'outputs', []):
                        if output.output_type == "stream":
                            cell_data["output"].append(output.text)
                        elif output.output_type == "execute_result":
                            cell_data["output"].append(output.data.get("text/plain", ""))
                        elif output.output_type == "error":
                            traceback = "\n".join(output.traceback)
                            cell_data["output"].append(f"Error: {output.ename}\n{traceback}")
                
                processed_cells.append(cell_data)

            return render_template(
                "notebook_editor_two.html", 
                cells=processed_cells,
                notebook_name=file.filename
            )

        except Exception as e:
            current_app.logger.error(f"Error processing notebook: {str(e)}", exc_info=True)
            return f"Error processing notebook: {str(e)}", 500

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

@kaggle_bp.route("/sequential_two")
@login_required
def sequential_notebook_two():
    notebook = UserNotebook.query.filter_by(user_id=current_user.id, is_sequential=True).order_by(UserNotebook.updated_at.desc()).first()
    return render_template("sequential_notebook_two.html", notebook=notebook)


@kaggle_bp.route("/datasets")
@login_required
def list_datasets():
    from models import DatasetMeta
    files = DatasetMeta.query.order_by(DatasetMeta.uploaded_at.desc()).all()

    # Add preview field to each file
    for f in files:
        try:
            path = os.path.join("datasets", f.filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as infile:
                lines = infile.readlines()
                f.preview = "".join(lines[:10])  # First 10 lines as preview
        except Exception as e:
            f.preview = f"⚠️ Preview not available: {str(e)}"

    return render_template("list_datasets.html", files=files)

# @kaggle_bp.route("/datasets")
# @login_required
# def list_datasets():
#     from models import DatasetMeta
#     files = DatasetMeta.query.order_by(DatasetMeta.uploaded_at.desc()).all()
#     return render_template("list_datasets.html", files=files)

@kaggle_bp.route("/sequential/list")
@login_required
def list_sequential_notebooks():
    notebooks = UserNotebook.query.filter_by(user_id=current_user.id, is_sequential=True).order_by(UserNotebook.updated_at.desc()).all()
    return jsonify([
        {
            "id": nb.id,
            "name": nb.name,
            "updated_at": nb.updated_at.isoformat() if nb.updated_at else None
        } for nb in notebooks
    ])

@kaggle_bp.route("/save_user_notebook", methods=["POST"])
@login_required
def save_user_notebook():
    data = request.get_json()
    name = data.get("name", "Untitled Notebook")
    notebook_data = data.get("notebook", [])
    
    # Create new notebook entry
    new_entry = UserNotebook(
        user_id=current_user.id,
        name=name,
        content=json.dumps(notebook_data)
    )
    db.session.add(new_entry)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": "Notebook saved successfully",
        "id": new_entry.id,
        "name": name
    })

@kaggle_bp.route("/notebook/update/<int:notebook_id>", methods=["POST"])
@login_required
def update_notebook(notebook_id):
    notebook = UserNotebook.query.filter_by(
        id=notebook_id, 
        user_id=current_user.id
    ).first()
    
    if not notebook:
        return jsonify({
            "success": False,
            "message": "Notebook not found"
        }), 404
    
    data = request.get_json()
    notebook.content = json.dumps(data.get("notebook", []))
    
    if "name" in data:
        notebook.name = data["name"]
    
    db.session.commit()
    return jsonify({
        "success": True,
        "message": "Notebook updated successfully",
        "id": notebook_id,
        "name": notebook.name
    })


@kaggle_bp.route("/my_notebooks")
@login_required
def my_notebooks():
    notebooks = UserNotebook.query.filter_by(user_id=current_user.id).order_by(UserNotebook.updated_at.desc()).all()
    
    # Add plotting info to each notebook
    notebooks_with_plotting = []
    for nb in notebooks:
        try:
            content = json.loads(nb.content) if isinstance(nb.content, str) else nb.content
            is_plotting = is_plotting_notebook(content)
        except:
            is_plotting = False
            
        notebooks_with_plotting.append((nb, is_plotting))
    
    return render_template("notebooks_two.html", notebooks=notebooks_with_plotting)


@kaggle_bp.route("/runner")
@login_required
def runner():
    return render_template("runner.html")


@kaggle_bp.route("/edit_notebook/<int:notebook_id>")
@login_required
def edit_saved_notebook(notebook_id):
    nb = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not nb:
        return "Notebook not found", 404

    try:
        notebook_content = json.loads(nb.content)
        if not isinstance(notebook_content, list):
            notebook_content = []
    except Exception as e:
        print("⚠️ JSON parsing error:", e)
        notebook_content = []

    # Check both the content and query parameter
    force_plotting = request.args.get('editor') == 'plotting'
    template = "editor_two.html" if force_plotting or is_plotting_notebook(notebook_content) else "editor.html"
    
    return render_template(
    template,
    saved_cells=notebook_content,  # this must contain all code/markdown cells
    notebook_id=nb.id,
    notebook_name=nb.name
)



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

    return render_template("notebook_editor_two.html", cells=code_cells)



@kaggle_bp.route("/run", methods=["GET", "POST"])
def run():
    import requests as rq

    if request.method == "POST":
        file = request.files.get("files")
        if not file:
            return "❌ No file uploaded", 400

        code = file.read().decode("utf-8")

        try: 
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

# @kaggle_bp.route("/notebook/update/<int:notebook_id>", methods=["POST"])
# @login_required
# def update_notebook(notebook_id):
#     try:
#         data = request.get_json()
#         new_cells = data.get("notebook", [])
#         new_name = data.get("name")

#         notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
#         if not notebook:
#             return "Notebook not found or unauthorized", 404

#         if new_name:
#             notebook.name = new_name
#         if new_cells:
#             notebook.content = json.dumps(new_cells)

#         db.session.commit()
#         return "✅ Notebook updated", 200

#     except Exception as e:
#         print("❌ Update error:", e)
#         return f"❌ Failed to update notebook: {str(e)}", 500
    
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

@kaggle_bp.route("/editor/three", methods=["GET"])
def monaco_editor_three():
    return render_template(
        "editor_three.html",
        saved_cells=[],
        notebook_name="Untitled Notebook",
        notebook_id="null"
    )


@kaggle_bp.route("/jup")
def jup():
    return render_template("jup_lite.html")

@kaggle_bp.route("/rendered/<filename>")
def rendered_file(filename):
    return send_from_directory("rendered_notebooks", filename)

@kaggle_bp.route("/sequential/<int:notebook_id>")
@login_required
def open_sequential_notebook(notebook_id):
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first_or_404()
    
    try:
        # Parse the content to ensure it's valid JSON
        content = json.loads(notebook.content)
        # If content is a string (old format), convert to proper structure
        if isinstance(content, str):
            content = [{"type": "code", "content": content}]
    except json.JSONDecodeError:
        # Handle case where content isn't valid JSON
        content = [{"type": "code", "content": notebook.content}]
    except Exception as e:
        print(f"Error parsing notebook content: {e}")
        content = []
    
    return render_template(
        "sequential_notebook_two.html",
        notebook=notebook,
        notebook_content=json.dumps(content)  # Properly formatted JSON string
    )

# @kaggle_bp.route("/notebook/update/<int:notebook_id>", methods=["POST"])
# @login_required
# def update_notebook(notebook_id):
#     notebook = UserNotebook.query.filter_by(
#         id=notebook_id, 
#         user_id=current_user.id
#     ).first()
    
#     if not notebook:
#         return jsonify({
#             "success": False,
#             "message": "Notebook not found"
#         }), 404
    
#     data = request.get_json()
#     notebook.content = json.dumps(data.get("notebook", []))
    
#     if "name" in data:
#         notebook.name = data["name"]
    
#     db.session.commit()
#     return jsonify({
#         "success": True,
#         "message": "Notebook updated successfully"
#     })

@kaggle_bp.route("/notebook/delete/<int:notebook_id>", methods=["POST"])
@login_required
def delete_notebook(notebook_id):
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first()
    if not notebook:
        return jsonify({"success": False, "message": "Notebook not found"}), 404
    
    db.session.delete(notebook)
    db.session.commit()
    return jsonify({"success": True, "message": "Notebook deleted successfully"})

@kaggle_bp.route("/code", methods=["GET"])
def code():
    return render_template('code.html')
@kaggle_bp.route("/notebook/rename/<int:notebook_id>", methods=["POST"])
@login_required
def rename_notebook(notebook_id):
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Content-Type must be application/json"
            }), 400

        data = request.get_json()
        new_name = data.get("new_name", "").strip()
        
        if not new_name:
            return jsonify({
                "success": False,
                "message": "Name cannot be empty"
            }), 400

        notebook = UserNotebook.query.filter_by(
            id=notebook_id, 
            user_id=current_user.id
        ).first()
        
        if not notebook:
            return jsonify({
                "success": False,
                "message": "Notebook not found"
            }), 404

        notebook.name = new_name
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Notebook renamed successfully",
            "new_name": new_name
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error renaming notebook: {str(e)}"
        }), 500
    

@kaggle_bp.route("/evaluate_plot", methods=["POST"])
def evaluate_plot():
    try:
        data = request.get_json()
        code = data.get("code", "")
        
        # Redirect stdout to capture print statements
        buffer = BytesIO()
        
        # Check if the code might generate a plot
        if any(keyword in code for keyword in ['plt.', 'plot(', 'figure(', 'show(']):
            # Create a new figure
            plt.figure()
            
            # Execute the code
            exec(code, {'plt': plt}, {})
            
            # Save the plot to buffer
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            plt.close()
            img_buffer.seek(0)
            
            return jsonify({
                "success": True,
                "plot": base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            })
        else:
            # Regular code execution
            import io
            import contextlib
            
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, {}, {})
                
            return jsonify({
                "success": True,
                "result": output.getvalue() or "✅ Execution successful"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "result": f"❌ Error: {str(e)}"
        })
    
@kaggle_bp.route("/editor_two", methods=["GET"])
def monaco_editor_two():
    return render_template(
        "editor_two.html",
        saved_cells=[],
        notebook_name="Untitled Notebook",
        notebook_id="null"
    )

@kaggle_bp.route("/notebook/sell/<int:notebook_id>", methods=["POST"])
@login_required
def sell_notebook(notebook_id):
    price = float(request.form.get("price", 0.0))
    notebook = UserNotebook.query.get_or_404(notebook_id)

    if notebook.user_id != current_user.id:
        return "Unauthorized", 403

    notebook.is_for_sale = True
    notebook.price = price
    db.session.commit()
    return redirect(url_for("app.my_notebooks"))

@kaggle_bp.route("/marketplace")
def notebook_marketplace():
    listings = UserNotebook.query.filter_by(is_for_sale=True).order_by(UserNotebook.updated_at.desc()).all()
    return render_template("marketplace.html", listings=listings)

@kaggle_bp.route("/notebook/buy/<int:notebook_id>", methods=["POST"])
@login_required
def buy_notebook(notebook_id):
    from models import Users, WalletDB, db

    notebook = UserNotebook.query.get_or_404(notebook_id)
    buyer = current_user
    seller = Users.query.get_or_404(notebook.user_id)
    
    seller_wallet = WalletDB.query.filter_by(address=seller.username).first()
    buyer_wallet = WalletDB.query.filter_by(address=buyer.username).first()

    # Check sale validity and funds
    if not notebook.is_for_sale:
        return "❌ Notebook is not for sale", 400
    if buyer_wallet.coins < notebook.price:
        return "❌ Insufficient coins", 402

    # Transfer coins
    buyer_wallet.coins -= notebook.price
    seller_wallet.coins += notebook.price

    # Copy notebook to buyer
    new_copy = UserNotebook(
        user_id=buyer.id,
        name=f"Purchased: {notebook.name}",
        content=notebook.content,
        is_for_sale=False,
        price=0.0
    )

    db.session.add(new_copy)
    db.session.commit()

    return redirect(url_for("app.my_notebooks"))

@kaggle_bp.route("/notebook/unlist/<int:notebook_id>", methods=["POST"])
@login_required
def unlist_notebook(notebook_id):
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first_or_404()

    notebook.is_for_sale = False
    notebook.price = 0.0
    db.session.commit()

    return redirect(url_for("app.my_notebooks"))


@kaggle_bp.route("/notebook/convert_to_html/<int:notebook_id>", methods=["GET"])
@login_required
def convert_notebook_to_html(notebook_id):
    """Convert a notebook to HTML format"""
    from nbconvert import HTMLExporter
    import nbformat
    from io import BytesIO
    
    # Get the notebook from database
    notebook = UserNotebook.query.filter_by(id=notebook_id, user_id=current_user.id).first_or_404()
    
    try:
        # Convert notebook content to standard ipynb format
        notebook_content = json.loads(notebook.content)
        
        # Create a new notebook structure
        nb = nbformat.v4.new_notebook()
        nb.cells = []
        
        for cell in notebook_content:
            if cell.get("type") == "markdown":
                nb.cells.append(nbformat.v4.new_markdown_cell(cell.get("content", "")))
            else:  # default to code cell
                nb.cells.append(nbformat.v4.new_code_cell(cell.get("content", "")))
        
        # Convert to HTML
        html_exporter = HTMLExporter()
        html_exporter.exclude_input_prompt = True
        html_exporter.exclude_output_prompt = True
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        # Create in-memory file for download
        buffer = BytesIO()
        buffer.write(body.encode('utf-8'))
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{notebook.name.replace(' ', '_')}.html",
            mimetype="text/html"
        )
        
    except Exception as e:
        current_app.logger.error(f"Conversion error: {str(e)}")
        return f"Error converting notebook: {str(e)}", 500

@kaggle_bp.route("/list_user_notebooks")
@login_required
def list_user_notebooks():
    notebooks = UserNotebook.query.filter_by(user_id=current_user.id).order_by(UserNotebook.updated_at.desc()).all()
    return jsonify([
        {"id": nb.id, "name": nb.name, "updated_at": nb.updated_at.isoformat()} for nb in notebooks
    ])

@kaggle_bp.route("/editor/<int:notebook_id>")
@login_required
def open_saved_notebook(notebook_id):
    notebook = UserNotebook.query.get_or_404(notebook_id)
    cells = json.loads(notebook.content)  # or notebook.cells depending on your model
    return render_template("editor.html",
                           saved_cells=cells,
                           notebook_id=notebook.id,
                           notebook_name=notebook.name)
