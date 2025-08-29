# jobs_blueprint.py
from flask import Blueprint, request, jsonify, render_template, abort
from models import db
from models_async import AsyncJob
from executor_service import submit_job

jobs_bp = Blueprint("jobs_bp", __name__, template_folder="templates")

@jobs_bp.route("/api/create", methods=["POST"])
def api_create_job():
    data = request.get_json(force=True, silent=True) or {}
    task = data.get("task", "gpu.heavy")
    payload = data.get("payload", {})
    prefer_gpu = bool(data.get("prefer_gpu", True))
    gpu_id = data.get("gpu_id")
    user_id = data.get("user_id")
    notify_channel = data.get("notify_channel")  # optional custom room

    try:
        job = submit_job(task, payload, user_id, prefer_gpu, gpu_id, notify_channel)
        return jsonify({
            "job_id": job.id,
            "celery_id": job.celery_id,
            "queue": job.queue,
            "gpu_id": job.gpu_id,
            "status": job.status,
        }), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@jobs_bp.route("/api/<int:job_id>", methods=["GET"])
def api_job_status(job_id):
    job = AsyncJob.query.get(job_id)
    if not job:
        abort(404)
    return jsonify({
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result_json and job.result_json,
        "error": job.error,
        "queue": job.queue,
        "gpu_id": job.gpu_id,
        "task_name": job.task_name,
    })

@jobs_bp.route("/dashboard", methods=["GET"])
def jobs_dashboard():
    return render_template("jobs_dashboard.html")
