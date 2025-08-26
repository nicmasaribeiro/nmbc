# collab.py
from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
from flask_socketio import join_room, leave_room, emit
from datetime import datetime
from models import db, UserNotebook
import json

# IMPORTANT: socketio must be created in your app entry (see Section 3)
from app import socketio  # or wherever you initialize SocketIO

collab_bp = Blueprint("collab", __name__, template_folder="templates")

# In-memory state. Replace with Redis for multi-worker deployments.
# Structure: { notebook_id: {"cells": [...], "version": int} }
NOTEBOOK_STATE = {}


def _load_notebook_state(notebook_id):
    """Load state from DB into memory (or return existing)."""
    state = NOTEBOOK_STATE.get(notebook_id)
    if state is None:
        nb = UserNotebook.query.get(notebook_id)
        cells = []
        if nb and nb.content:
            try:
                cells = json.loads(nb.content)
            except Exception:
                cells = []
        NOTEBOOK_STATE[notebook_id] = {"cells": cells, "version": 1}
    return NOTEBOOK_STATE[notebook_id]

@collab_bp.route("/collab/new", methods=["POST"])
@login_required
def create_collab_notebook():
    """Create a new collaborative notebook."""
    data = request.get_json()
    name = data.get("name", "Untitled Collaborative Notebook")
    cells = data.get("notebook", [])

    # Create DB entry
    notebook = UserNotebook(
        user_id=current_user.id,
        name=name,
        content=json.dumps(cells),
        tags="collaborative"
    )
    db.session.add(notebook)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Collaborative notebook created",
        "id": notebook.id,
        "name": notebook.name
    })
@collab_bp.route("/collab/<int:notebook_id>")
@login_required
def collab_editor(notebook_id):
    nb = UserNotebook.query.filter_by(id=notebook_id).first()
    if not nb:
        return "Notebook not found", 404

    state = _load_notebook_state(notebook_id)
    return render_template(
        "collab_editor.html",
        notebook_id=notebook_id,
        username=current_user.username,
        initial_cells=json.dumps(state["cells"]),
        initial_version=state["version"],
        notebook_name=nb.name or f"Notebook {notebook_id}",
    )


# -------- Socket events --------

@socketio.on("join_notebook")
def ws_join(data):
    notebook_id = int(data["notebook_id"])
    username = data.get("username", "guest")
    room = f"notebook_{notebook_id}"

    _load_notebook_state(notebook_id)
    join_room(room)

    # Send current state to the joiner only
    state = NOTEBOOK_STATE[notebook_id]
    emit("state_sync", {"cells": state["cells"], "version": state["version"]})
    # Tell others that someone joined
    emit("presence", {"user": username, "event": "join"}, to=room, include_self=False)


@socketio.on("leave_notebook")
def ws_leave(data):
    notebook_id = int(data["notebook_id"])
    username = data.get("username", "guest")
    room = f"notebook_{notebook_id}"
    leave_room(room)
    emit("presence", {"user": username, "event": "leave"}, to=room, include_self=False)


@socketio.on("cell_update")
def ws_cell_update(data):
    """
    Payload: {
      notebook_id, index, content, version, username
    }
    """
    notebook_id = int(data["notebook_id"])
    idx = int(data["index"])
    content = data.get("content", "")
    client_version = int(data.get("version", 0))
    username = data.get("username", "guest")

    state = _load_notebook_state(notebook_id)

    # Ensure cell exists
    while len(state["cells"]) <= idx:
        state["cells"].append({"type": "code", "content": ""})

    # Simple last-write-wins using version bump
    if client_version >= state["version"]:
        state["cells"][idx]["content"] = content
        state["version"] += 1

    room = f"notebook_{notebook_id}"
    emit(
        "cell_applied",
        {"index": idx, "content": content, "version": state["version"], "user": username},
        to=room,
        include_self=False,
    )


@socketio.on("add_cell")
def ws_add_cell(data):
    """
    Payload: { notebook_id, after, type }
    """
    notebook_id = int(data["notebook_id"])
    after = int(data.get("after", -1))
    cell_type = data.get("type", "code")

    state = _load_notebook_state(notebook_id)
    insert_at = max(0, after + 1)
    state["cells"].insert(insert_at, {"type": cell_type, "content": ""})
    state["version"] += 1

    room = f"notebook_{notebook_id}"
    emit(
        "cell_inserted",
        {"index": insert_at, "cell": state["cells"][insert_at], "version": state["version"]},
        to=room,
    )


@socketio.on("delete_cell")
def ws_delete_cell(data):
    """
    Payload: { notebook_id, index }
    """
    notebook_id = int(data["notebook_id"])
    idx = int(data["index"])

    state = _load_notebook_state(notebook_id)
    if 0 <= idx < len(state["cells"]):
        state["cells"].pop(idx)
        state["version"] += 1
        room = f"notebook_{notebook_id}"
        emit("cell_deleted", {"index": idx, "version": state["version"]}, to=room)


@socketio.on("cursor")
def ws_cursor(data):
    """
    Payload: { notebook_id, index, pos, username }
    """
    notebook_id = int(data["notebook_id"])
    room = f"notebook_{notebook_id}"
    emit(
        "cursor",
        {"index": data["index"], "pos": data["pos"], "user": data.get("username", "guest")},
        to=room,
        include_self=False,
    )


@socketio.on("save_notebook")
def ws_save(data):
    """
    Payload: { notebook_id }
    """
    notebook_id = int(data["notebook_id"])
    state = _load_notebook_state(notebook_id)

    nb = UserNotebook.query.get(notebook_id)
    if not nb:
        emit("saved", {"ok": False, "message": "Notebook not found"})
        return

    nb.content = json.dumps(state["cells"])
    nb.updated_at = datetime.utcnow()
    db.session.commit()
    emit("saved", {"ok": True, "version": state["version"]})
