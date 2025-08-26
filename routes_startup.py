# routes_startup.py
from flask import Blueprint, render_template, current_app

startup_bp = Blueprint("startup", __name__, template_folder="templates")

@startup_bp.route("/")
def startup_home():
    # Safely provide counts/summaries if your models are available; fall back to 0
    metrics = {
        "notebooks": 0,
        "datasets": 0,
        "markets": 0,
        "jobs": 0,
        "users": 0,
    }
    try:
        from models import UserNotebook, DatasetMeta, PredictionMarket, JobQueue, Users
        metrics["notebooks"] = getattr(UserNotebook.query, "count", lambda: 0)()
        metrics["datasets"]  = getattr(DatasetMeta.query, "count", lambda: 0)()
        metrics["markets"]   = getattr(PredictionMarket.query, "count", lambda: 0)()
        metrics["jobs"]      = getattr(JobQueue.query, "count", lambda: 0)()
        metrics["users"]     = getattr(Users.query, "count", lambda: 0)()
    except Exception:
        pass

    app_name = getattr(current_app.config, "APP_NAME", "Nmbcy")
    version  = getattr(current_app.config, "APP_VERSION", "v0.1")
    return render_template("startup.html", app_name=app_name, version=version, metrics=metrics)
