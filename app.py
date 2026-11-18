"""
Study Room Scheduler - small web app (Flask).
Run:  python app.py      then open  http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request

import scheduler

app = Flask(__name__)


def to_number(value, name):
    """Convert a value from the web form to a number or raise a clear error."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")


@app.route("/")
def index():
    return render_template("index.html", names=scheduler.NAMES, scores=scheduler.DEFAULT_SCORES,
                           defaults=scheduler.Settings())


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/decide", methods=["POST"])
def api_decide():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Please send JSON data."), 400
    try:
        report = data.get("report")
        scores = data.get("scores", scheduler.DEFAULT_SCORES)
        if not isinstance(scores, list):
            raise ValueError("scores must be a list")
        scores = [to_number(v, "score") for v in scores]
        d = scheduler.Settings()
        settings = scheduler.Settings(
            prior=to_number(data.get("prior", d.prior), "prior"),
            p_free_if_available=to_number(data.get("p_free_if_available", d.p_free_if_available), "p_free_if_available"),
            p_free_if_taken=to_number(data.get("p_free_if_taken", d.p_free_if_taken), "p_free_if_taken"))
        restarts = bool(data.get("restarts", False))
        result = scheduler.decide(report, scores, settings, restarts)
        result["evaluation"] = scheduler.evaluate(scores, settings, restarts)
        return jsonify(result)
    except ValueError as error:
        return jsonify(error=str(error)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
