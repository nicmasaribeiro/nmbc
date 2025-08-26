# notebook_state.py
import json
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from models import db, UserNotebook

class NotebookStateManager:
    """
    Manages collaborative state per notebook with optimistic concurrency (cell-level).
    Storage:
      - If redis_client provided: keys "nb:{id}:version" (int), "nb:{id}:cells" (json str)
      - Otherwise in-memory dict.
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._mem: Dict[int, Dict[str, Any]] = {}
        self._locks: Dict[int, threading.Lock] = {}

    # ---------- Low-level storage ----------
    def _get_lock(self, notebook_id: int) -> threading.Lock:
        if notebook_id not in self._locks:
            self._locks[notebook_id] = threading.Lock()
        return self._locks[notebook_id]

    def _load_from_db(self, notebook_id: int) -> Tuple[List[Dict[str, Any]], int]:
        nb = UserNotebook.query.get(notebook_id)
        if not nb:
            return [], 1
        try:
            cells = json.loads(nb.content) if nb.content else []
        except Exception:
            cells = []
        return cells, 1

    def _read_store(self, notebook_id: int) -> Tuple[List[Dict[str, Any]], int]:
        if self.redis:
            v = self.redis.get(f"nb:{notebook_id}:version")
            c = self.redis.get(f"nb:{notebook_id}:cells")
            if v is None or c is None:
                cells, version = self._load_from_db(notebook_id)
                pipe = self.redis.pipeline()
                pipe.set(f"nb:{notebook_id}:version", version)
                pipe.set(f"nb:{notebook_id}:cells", json.dumps(cells))
                pipe.execute()
                return cells, version
            return json.loads(c), int(v)
        # memory fallback
        state = self._mem.get(notebook_id)
        if not state:
            cells, version = self._load_from_db(notebook_id)
            self._mem[notebook_id] = {"cells": cells, "version": version, "last_saved_version": 0}
            return cells, version
        return state["cells"], state["version"]

    def _write_store(self, notebook_id: int, cells: List[Dict[str, Any]], version: int):
        if self.redis:
            pipe = self.redis.pipeline()
            pipe.set(f"nb:{notebook_id}:version", version)
            pipe.set(f"nb:{notebook_id}:cells", json.dumps(cells))
            pipe.execute()
            return
        state = self._mem.setdefault(notebook_id, {"cells": [], "version": 1, "last_saved_version": 0})
        state["cells"] = cells
        state["version"] = version

    # ---------- Public API ----------
    def get_state(self, notebook_id: int) -> Dict[str, Any]:
        cells, version = self._read_store(notebook_id)
        return {"cells": cells, "version": version}

    def insert_cell(self, notebook_id: int, at_index: int, cell_type: str = "code") -> Dict[str, Any]:
        with self._get_lock(notebook_id):
            cells, version = self._read_store(notebook_id)
            at = max(0, min(at_index, len(cells)))
            cells = cells.copy()
            cells.insert(at, {"type": cell_type, "content": ""})
            version += 1
            self._write_store(notebook_id, cells, version)
            return {"index": at, "cell": cells[at], "version": version}

    def delete_cell(self, notebook_id: int, index: int) -> Optional[Dict[str, Any]]:
        with self._get_lock(notebook_id):
            cells, version = self._read_store(notebook_id)
            if 0 <= index < len(cells):
                cells = cells.copy()
                cells.pop(index)
                version += 1
                self._write_store(notebook_id, cells, version)
                return {"index": index, "version": version}
            return None

    def apply_cell_update(self, notebook_id: int, index: int, content: str, base_version: int) -> Dict[str, Any]:
        """
        Optimistic concurrency: update applied if base_version >= current_version.
        If stale, return {"accepted": False, "version": current_version}.
        """
        with self._get_lock(notebook_id):
            cells, version = self._read_store(notebook_id)

            # Ensure cell exists
            while len(cells) <= index:
                cells.append({"type": "code", "content": ""})

            if base_version < version:
                return {"accepted": False, "version": version}

            cells = cells.copy()
            cells[index] = {**cells[index], "content": content}
            version += 1
            self._write_store(notebook_id, cells, version)
            return {"accepted": True, "index": index, "content": content, "version": version}

    def save_to_db(self, notebook_id: int) -> Dict[str, Any]:
        """Persist current state to DB."""
        with self._get_lock(notebook_id):
            cells, version = self._read_store(notebook_id)
            nb = UserNotebook.query.get(notebook_id)
            if not nb:
                return {"ok": False, "message": "Notebook not found"}

            nb.content = json.dumps(cells)
            db.session.commit()
            return {"ok": True, "version": version}

# Singleton accessor
_manager: Optional[NotebookStateManager] = None

def get_notebook_manager(redis_client=None) -> NotebookStateManager:
    global _manager
    if _manager is None:
        _manager = NotebookStateManager(redis_client=redis_client)
    return _manager
