"""
Executor Service — Async Job Queue with Optional GPU Support and Notifications

Drop-in service that lets you submit CPU/GPU jobs in a fire‑and‑forget way and
receive notifications when they finish. Designed for Flask apps (works without
Flask too). Optional Socket.IO push updates.

Key features
- In‑memory priority queues for CPU and GPU jobs
- Per‑GPU serialized workers (no accidental over‑subscription)
- CPU thread pool for parallel jobs
- Fire‑and‑forget submit() returns immediately with job_id
- Pluggable Notifier (Socket.IO / print / custom)
- Task registry (submit by name remotely via API)
- Safe torch.cuda handling (if PyTorch is available)
- Introspect status/result/errors; cancel queued jobs

Usage (minimal)
---------------
from executor_service import ExecutorService, PrintNotifier
svc = ExecutorService()
job_id = svc.submit(lambda x: x*x, 12)  # CPU job

GPU (if torch is available)
---------------------------
job_id = svc.submit(task_fn,  payload, job_type="gpu")
# or choose device explicitly: job_type="gpu", device="cuda:0"

Flask blueprint (optional)
--------------------------
from executor_service import create_jobs_blueprint, ExecutorService
svc = ExecutorService(socketio=socketio)  # if you use Flask‑SocketIO
jobs_bp = create_jobs_blueprint(svc)
app.register_blueprint(jobs_bp, url_prefix="/api/jobs")

Socket.IO notifications
-----------------------
On finish/error/progress it emits "job_update" to room=f"user:{user_id}" (if provided)
with payload { job_id, status, result, error, progress }.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
import queue
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

# Optional: PyTorch for GPU device context
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

__all__ = [
    "ExecutorService",
    "TaskRegistry",
    "BaseNotifier",
    "PrintNotifier",
    "SocketIONotifier",
    "create_jobs_blueprint",
]

# -----------------------------
# Data structures
# -----------------------------

JobStatus = str  # "queued"|"running"|"done"|"error"|"canceled"

@dataclass
class Job:
    job_id: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    job_type: str = "cpu"  # "cpu" | "gpu"
    device: Optional[str] = None  # e.g. "cuda:0"
    priority: int = 100  # lower is earlier

    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    result: Any = None
    error: Optional[str] = None
    future: Optional[Future] = None  # for CPU worker pool cancellation

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Future isn't JSON serializable and not interesting to clients
        d.pop("future", None)
        return d

# -----------------------------
# Notifiers
# -----------------------------

class BaseNotifier:
    def notify(self, event: str, payload: Dict[str, Any], user_id: Optional[str] = None) -> None:
        raise NotImplementedError

class PrintNotifier(BaseNotifier):
    def notify(self, event: str, payload: Dict[str, Any], user_id: Optional[str] = None) -> None:
        logging.info("[Notifier:%s][user=%s] %s", event, user_id, payload)

class SocketIONotifier(BaseNotifier):
    """Flask-SocketIO notifier.

    Emits to room f"user:{user_id}" if user_id is provided, else broadcasts.
    """
    def __init__(self, socketio) -> None:
        self.socketio = socketio

    def notify(self, event: str, payload: Dict[str, Any], user_id: Optional[str] = None) -> None:
        room = f"user:{user_id}" if user_id else None
        try:
            self.socketio.emit("job_update", {"event": event, **payload}, to=room)
        except Exception as e:
            logging.exception("SocketIO notify failed: %s", e)

# -----------------------------
# Task Registry (optional)
# -----------------------------

class TaskRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        with self._lock:
            self._tasks[name] = fn

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tasks.pop(name, None)

    def get(self, name: str) -> Callable[..., Any]:
        with self._lock:
            if name not in self._tasks:
                raise KeyError(f"Task '{name}' not found")
            return self._tasks[name]

# -----------------------------
# Executor Service
# -----------------------------

class ExecutorService:
    def __init__(
        self,
        *,
        max_cpu_workers: int = max(1, (0 if False else 4)),  # sensible default for webapp
        gpu_devices: Optional[List[str]] = None,
        notifier: Optional[BaseNotifier] = None,
        socketio: Any = None,
        enable_autocast: bool = True,
    ) -> None:
        """Create the service.

        Args:
            max_cpu_workers: size of CPU worker threadpool.
            gpu_devices: list like ["cuda:0", "cuda:1"]. If None, auto-detect when torch is present.
            notifier: if None and socketio provided, uses SocketIONotifier; else PrintNotifier.
            socketio: Flask-SocketIO instance for push notifications.
            enable_autocast: enable torch autocast for gpu jobs where possible.
        """
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)

        # Notifier
        if notifier is not None:
            self.notifier = notifier
        elif socketio is not None:
            self.notifier = SocketIONotifier(socketio)
        else:
            self.notifier = PrintNotifier()

        # Task registry
        self.tasks = TaskRegistry()

        # Job storage
        self._jobs: Dict[str, Job] = {}
        self._jobs_lock = threading.Lock()

        # Queues
        self._cpu_queue: "queue.PriorityQueue[Tuple[int, float, str]]" = queue.PriorityQueue()
        # Per-device GPU queues
        self._gpu_queues: Dict[str, "queue.PriorityQueue[Tuple[int, float, str]]"] = {}

        # Workers
        self._cpu_pool = ThreadPoolExecutor(max_workers=max_cpu_workers, thread_name_prefix="cpu-job")
        self._stop_event = threading.Event()

        # Detect GPUs
        self.enable_autocast = enable_autocast and _TORCH_AVAILABLE
        self.gpu_devices: List[str] = []
        if gpu_devices is not None:
            self.gpu_devices = gpu_devices
        elif _TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        # Init GPU queues and worker threads (1 thread per device to serialize jobs on that GPU)
        self._gpu_workers: Dict[str, threading.Thread] = {}
        for dev in self.gpu_devices:
            q: "queue.PriorityQueue[Tuple[int, float, str]]" = queue.PriorityQueue()
            self._gpu_queues[dev] = q
            t = threading.Thread(target=self._gpu_worker_loop, args=(dev,), daemon=True, name=f"gpu-worker-{dev}")
            t.start()
            self._gpu_workers[dev] = t
            self.log.info("Started GPU worker for %s", dev)

        # CPU dispatcher thread (pulls from the priority queue and submits to pool)
        self._cpu_dispatcher = threading.Thread(target=self._cpu_dispatch_loop, daemon=True, name="cpu-dispatcher")
        self._cpu_dispatcher.start()

    # -----------------------------
    # Public API
    # -----------------------------

    def register_task(self, name: str, fn: Callable[..., Any]) -> None:
        self.tasks.register(name, fn)

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        user_id: Optional[str] = None,
        job_type: str = "cpu",
        device: Optional[str] = None,
        priority: int = 100,
        **kwargs: Any,
    ) -> str:
        """Fire‑and‑forget submission. Returns a job_id immediately."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            func=func,
            args=args,
            kwargs=kwargs,
            user_id=user_id,
            job_type=job_type,
            device=device,
            priority=int(priority),
        )
        with self._jobs_lock:
            self._jobs[job_id] = job

        if job_type == "gpu":
            target = device or self._pick_gpu_device()
            if target is None:
                job.status = "error"
                job.error = "No GPU device available"
                self._finish_job(job)
                return job_id
            job.device = target
            self._gpu_queues[target].put((job.priority, job.created_at, job_id))
            self._notify("queued", job)
        else:
            self._cpu_queue.put((job.priority, job.created_at, job_id))
            self._notify("queued", job)
        return job_id

    def submit_task(
        self,
        task_name: str,
        *args: Any,
        user_id: Optional[str] = None,
        job_type: str = "cpu",
        device: Optional[str] = None,
        priority: int = 100,
        **kwargs: Any,
    ) -> str:
        fn = self.tasks.get(task_name)
        return self.submit(fn, *args, user_id=user_id, job_type=job_type, device=device, priority=priority, **kwargs)

    def get(self, job_id: str) -> Optional[Job]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        j = self.get(job_id)
        return j.to_dict() if j else None

    def result(self, job_id: str) -> Any:
        j = self.get(job_id)
        return getattr(j, "result", None) if j else None

    def cancel(self, job_id: str) -> bool:
        j = self.get(job_id)
        if not j:
            return False
        if j.status != "queued":
            return False
        # Try to remove from queue by rebuilding (PriorityQueue has no remove)
        if j.job_type == "gpu" and j.device in self._gpu_queues:
            self._rebuild_queue_without(self._gpu_queues[j.device], job_id)
        else:
            self._rebuild_queue_without(self._cpu_queue, job_id)
        j.status = "canceled"
        j.ended_at = time.time()
        self._notify("canceled", j)
        return True

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._jobs_lock:
            items = list(self._jobs.values())[-limit:]
            return [j.to_dict() for j in items]

    # -----------------------------
    # Internal loops
    # -----------------------------

    def _cpu_dispatch_loop(self) -> None:  # pragma: no cover
        self.log.info("CPU dispatcher started")
        while not self._stop_event.is_set():
            try:
                priority, created_at, job_id = self._cpu_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            job = self.get(job_id)
            if not job or job.status != "queued":
                continue
            job.status = "running"
            job.started_at = time.time()
            self._notify("started", job)
            # Submit to pool; attach callback
            future = self._cpu_pool.submit(self._run_job_callable, job)
            job.future = future
            future.add_done_callback(lambda f, jid=job.job_id: self._cpu_job_done_cb(jid, f))

    def _cpu_job_done_cb(self, job_id: str, fut: Future) -> None:  # pragma: no cover
        job = self.get(job_id)
        if not job:
            return
        try:
            res = fut.result()
            job.result = res
            job.status = "done"
        except Exception as e:  # noqa: BLE001
            job.error = repr(e)
            job.status = "error"
        finally:
            job.ended_at = time.time()
            self._finish_job(job)

    def _gpu_worker_loop(self, device: str) -> None:  # pragma: no cover
        self.log.info("GPU worker loop running for %s", device)
        q = self._gpu_queues[device]
        while not self._stop_event.is_set():
            try:
                priority, created_at, job_id = q.get(timeout=0.25)
            except queue.Empty:
                continue
            job = self.get(job_id)
            if not job or job.status != "queued":
                continue
            job.status = "running"
            job.started_at = time.time()
            self._notify("started", job)
            try:
                res = self._run_job_callable(job, gpu_device=device)
                job.result = res
                job.status = "done"
            except Exception as e:  # noqa: BLE001
                job.error = repr(e)
                job.status = "error"
            finally:
                job.ended_at = time.time()
                self._finish_job(job)

    # -----------------------------
    # Execution helpers
    # -----------------------------

    def _run_job_callable(self, job: Job, gpu_device: Optional[str] = None) -> Any:
        """Execute job.func with the proper device context.
        - For GPU jobs, set torch device and optionally autocast if available.
        - If the target function accepts a kwarg named 'device', we pass it.
        - For CPU jobs we just call func(*args, **kwargs).
        """
        kwargs = dict(job.kwargs)
        if gpu_device:
            # Make sure torch exists and device is valid
            if not _TORCH_AVAILABLE or not (torch and torch.cuda.is_available()):
                raise RuntimeError("GPU job submitted but torch.cuda is not available")

            # Select device
            # 'cuda:0' -> int 0
            try:
                torch.cuda.set_device(int(gpu_device.split(":")[1]))
            except Exception:  # tolerate unexpected format
                pass

            # Pass device to function if it declares it
            if "device" in job.func.__code__.co_varnames:
                kwargs.setdefault("device", gpu_device)

            if self.enable_autocast:
                # Use best effort dtype
                autocast_dtype = torch.float16 if torch.cuda.is_available() else None
                if autocast_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        return job.func(*job.args, **kwargs)
            # Fallback: plain call
            return job.func(*job.args, **kwargs)
        else:
            return job.func(*job.args, **kwargs)

    def _finish_job(self, job: Job) -> None:
        self._notify("finished" if job.status == "done" else job.status, job)

    def _notify(self, event: str, job: Job) -> None:
        try:
            self.notifier.notify(event, {"job": job.to_dict()}, user_id=job.user_id)
        except Exception as e:
            self.log.warning("Notifier failed: %s", e)

    def _pick_gpu_device(self) -> Optional[str]:
        # Choose device with the shortest queue (simple load balancing)
        if not self.gpu_devices:
            return None
        best_dev = None
        best_len = 1_000_000
        for dev, q in self._gpu_queues.items():
            qlen = q.qsize()
            if qlen < best_len:
                best_dev = dev
                best_len = qlen
        return best_dev

    @staticmethod
    def _rebuild_queue_without(q: "queue.PriorityQueue[Tuple[int, float, str]]", job_id: str) -> None:
        items: List[Tuple[int, float, str]] = []
        try:
            while True:
                items.append(q.get_nowait())
        except queue.Empty:
            pass
        for item in items:
            if item[2] != job_id:
                q.put(item)

# -----------------------------
# Flask Blueprint (optional)
# -----------------------------

def create_jobs_blueprint(service: ExecutorService):  # pragma: no cover - lightweight utility
    try:
        from flask import Blueprint, jsonify, request
    except Exception as e:  # Flask may not be present in some environments
        raise RuntimeError("Flask is required to create the jobs blueprint") from e

    bp = Blueprint("jobs", __name__)

    @bp.route("/", methods=["GET"])  # list
    def list_jobs_route():
        return jsonify({"jobs": service.list_jobs()})

    @bp.route("/<job_id>", methods=["GET"])  # detail
    def get_job_route(job_id: str):
        j = service.status(job_id)
        if not j:
            return jsonify({"error": "not found"}), 404
        return jsonify(j)

    @bp.route("/submit", methods=["POST"])  # submit by task name
    def submit_job_route():
        data = request.get_json(silent=True) or {}
        task = data.get("task")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        user_id = data.get("user_id")
        job_type = data.get("job_type", "cpu")
        device = data.get("device")
        priority = int(data.get("priority", 100))
        if not task:
            return jsonify({"error": "field 'task' is required"}), 400
        try:
            jid = service.submit_task(task, *args, user_id=user_id, job_type=job_type, device=device, priority=priority, **kwargs)
        except KeyError as e:
            return jsonify({"error": str(e)}), 404
        return jsonify({"job_id": jid})

    @bp.route("/<job_id>", methods=["DELETE"])  # cancel
    def cancel_job_route(job_id: str):
        ok = service.cancel(job_id)
        if not ok:
            return jsonify({"error": "cannot cancel"}), 400
        return jsonify({"ok": True})

    return bp

# -----------------------------
# Example built-in tasks (optional)
# -----------------------------

def _sleep_task(seconds: float) -> float:
    time.sleep(float(seconds))
    return seconds

# For GPU demo: moves a random tensor to the selected device and returns its norm

def _gpu_demo_task(size: int = 4, device: Optional[str] = None) -> float:
    if not _TORCH_AVAILABLE or not (torch and torch.cuda.is_available()):
        raise RuntimeError("Torch/CUDA not available")
    dev = torch.device(device or "cuda")
    x = torch.randn(size, size, device=dev)
    return float(x.norm().item())

# Convenience factory for quick testing

def create_default_service(socketio=None) -> ExecutorService:
    svc = ExecutorService(socketio=socketio)
    # Register a couple of utility tasks
    svc.register_task("sleep", _sleep_task)
    if _TORCH_AVAILABLE and (torch and torch.cuda.is_available()):
        svc.register_task("gpu_demo", _gpu_demo_task)
    return svc


# -----------------------------
# Quick integration snippet (Flask)
# -----------------------------
# from flask import Flask
# from flask_socketio import SocketIO
# from executor_service import create_default_service, create_jobs_blueprint
#
# socketio = SocketIO(cors_allowed_origins="*")
# app = Flask(__name__)
# socketio.init_app(app)
#
# exec_svc = create_default_service(socketio=socketio)
# app.register_blueprint(create_jobs_blueprint(exec_svc), url_prefix="/api/jobs")
#
# # Example: register an app-specific task
# def heavy_price_update(token: str) -> dict:
#     # ... long running logic here ...
#     return {"token": token, "ok": True}
#
# exec_svc.register_task("heavy_price_update", heavy_price_update)
#
# # Somewhere in your route when user clicks "Run":
# # jid = exec_svc.submit_task("heavy_price_update", token, user_id=str(current_user.id))
# # return jsonify({"job_id": jid})
#
# if __name__ == "__main__":
#     socketio.run(app, host="0.0.0.0", port=5000, debug=True)

# -----------------------------
# Client snippet (Socket.IO)
# -----------------------------
# <script src="https://cdn.socket.io/4.7.5/socket.io.min.js" crossorigin="anonymous"></script>
# <script>
#   const socket = io();
#   socket.on('connect', () => {
#     // Optionally join your user room after auth
#     // socket.emit('join', { room: 'user:{{ current_user.id }}' })
#   });
#   socket.on('job_update', (payload) => {
#     console.log('job_update', payload);
#     // payload.event: 'queued'|'started'|'finished'|'error'|'canceled'
#     // payload.job: {job_id, status, result, error, ...}
#     // Update UI e.g. progress list or toast notification
#   });
# </script>
