from extensions import socketio
from flask_login import current_user, login_required, login_user
from flask_socketio import emit
import concurrent.futures, contextlib, io, traceback, threading
from helper import *
EXEC_TIMEOUT = 6  # seconds
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

import io, contextlib, traceback, builtins

# ---- allowlist for imports (add what you need) ----
ALLOWED_MODULES = {
    "math", "statistics", "random",
    "json", "re", "itertools", "functools", "operator",
    "datetime", "decimal", "fractions",
    # uncomment only if you actually want these available:
    "numpy", "pandas","yfinance","sklearn"
}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restrict imports to a simple allowlist."""
    root = name.split(".", 1)[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)

# ---- builtins the code is allowed to see ----
SAFE_BUILTINS = {
    # common safe builtins
    "abs": builtins.abs,
    "all": builtins.all,
    "any": builtins.any,
    "bool": builtins.bool,
    "bytes": builtins.bytes,
    "callable": builtins.callable,
    "chr": builtins.chr,
    "complex": builtins.complex,
    "dict": builtins.dict,
    "dir": builtins.dir,
    "divmod": builtins.divmod,
    "enumerate": builtins.enumerate,
    "filter": builtins.filter,
    "float": builtins.float,
    "format": builtins.format,
    "frozenset": builtins.frozenset,
    "getattr": builtins.getattr,
    "hasattr": builtins.hasattr,
    "hash": builtins.hash,
    "hex": builtins.hex,
    "int": builtins.int,
    "isinstance": builtins.isinstance,
    "issubclass": builtins.issubclass,
    "iter": builtins.iter,
    "len": builtins.len,
    "list": builtins.list,
    "map": builtins.map,
    "max": builtins.max,
    "min": builtins.min,
    "next": builtins.next,
    "object": builtins.object,
    "oct": builtins.oct,
    "ord": builtins.ord,
    "pow": builtins.pow,
    "print": builtins.print,
    "range": builtins.range,
    "repr": builtins.repr,
    "reversed": builtins.reversed,
    "round": builtins.round,
    "set": builtins.set,
    "slice": builtins.slice,
    "sorted": builtins.sorted,
    "str": builtins.str,
    "sum": builtins.sum,
    "tuple": builtins.tuple,
    "zip": builtins.zip,
    # enable *controlled* importing
    "__import__": _safe_import,
}

def _run_code(code: str) -> str:
    buf = io.StringIO()

    # globals the user code sees
    g = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__main__",  # so `if __name__ == "__main__":` works
    }

    # locals for exec (kept separate in case you want to inspect later)
    l = {}

    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, g, l)
    except Exception:
        traceback.print_exc(file=buf)

    return buf.getvalue()


@socketio.on("run_cell")
@login_required
def ws_run_cell(data):
    """
    Payload: { notebook_id, index, code }
    """
    from flask import request
    notebook_id = int(data["notebook_id"])
    idx = int(data["index"])
    code = data.get("code", "")

    # permission: viewer cannot run
    nb, err = assert_access(notebook_id, current_user.id, "editor")
    if err:
        emit("cell_output", {"index": idx, "output": f"Permission denied: {err[1]}"}, to=request.sid)
        return

    room = f"nb:{notebook_id}"
    emit("cell_running", {"index": idx}, to=room)

    fut = _executor.submit(_run_code, code)
    try:
        out = fut.result(timeout=EXEC_TIMEOUT)
    except concurrent.futures.TimeoutError:
        out = f"[Timeout] Execution exceeded {EXEC_TIMEOUT}s"

    emit("cell_output", {"index": idx, "output": out}, to=room)
