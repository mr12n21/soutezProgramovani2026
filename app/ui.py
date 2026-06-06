"""
flask ui server
"""

import base64
import logging
import traceback
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from PIL import Image

from app.api import APIClient
from app.config import K, U
from app.map_parser import MP
from app.simulator import SIM

log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

state = {
    "api": None,
    "map": None,
    "map_parser": None,
    "vehicles": None,
    "simulator": None,
    "protocol": None,
}


def safe_api_call(func):
    """decorator"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": str(e), "type": type(e).__name__}), 400

    wrapper.__name__ = func.__name__
    return wrapper


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/init", methods=["POST", "GET"])
@safe_api_call
def api_init():
    """GET /settings"""
    state["api"] = APIClient(SERVER_URL, API_KEY)
    settings = state["api"].get_settings()

    state["vehicles"] = settings.get("vehicles", [])

    return jsonify(
        {
            "status": "ok",
            "session_id": state["api"].session_id,
            "vehicles": state["vehicles"],
        }
    )


@app.route("/api/load-map", methods=["POST", "GET"])
@safe_api_call
def api_load_map():
    """GET /map"""
    if not state["api"]:
        raise

    map_img = state["api"].get_map()
    state["map"] = map_img
    state["map_parser"] = MapParser(map_img)

    buf = BytesIO()
    map_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify(
        {
            "status": "ok",
            "size": list(map_img.size),
            "intersections": len(state["map_parser"].intersections),
            "image": b64,
        }
    )


@app.route("/api/load-requests", methods=["POST", "GET"])
@safe_api_call
def api_load_requests():
    """GET /requests"""
    if not state["api"]:
        raise ValueError("Zavolej /api/init")

    reqs = state["api"].get_requests()
    state["vehicles"] = reqs

    return jsonify({"status": "ok", "requests": reqs})


@app.route("/api/plan-routes", methods=["POST", "GET"])
@safe_api_call
def api_plan_routes():
    """trasa plan"""
    if not state["map_parser"] or not state["vehicles"]:
        raise

    state["simulator"] = Simulator(state["map_parser"], state["vehicles"])

    planned = sum(
        1 for v in state["simulator"].scheduler.vehicles if v["path"] is not None
    )
    total = len(state["simulator"].scheduler.vehicles)

    return jsonify({"status": "ok", "planned": planned, "total": total})


@app.route("/api/run-sim", methods=["POST", "GET"])
@safe_api_call
def api_run_sim():
    """simulace"""

    if not state["simulator"]:
        raise

    history = state["simulator"].run()

    return jsonify({"status": "ok", "steps": len(history), "history": history})


@app.route("/api/send-protocol", methods=["POST", "GET"])
@safe_api_call
def api_send_protocol():
    """POST protokol"""
    if not state["simulator"] or not state["api"]:
        raise

    protocol = state["simulator"].get_protocol(state["api"].session_id)
    state["protocol"] = protocol

    try:
        result = state["api"].post_protocol(protocol)
        return jsonify({"status": "ok", "server_response": result})
    except Exception as e:
        log.warning(f"Server communication failed: {e}")
        return jsonify(
            {
                "status": "offline",
                "message": "server",
                "protocol": protocol,
            }
        )


@app.route("/api/status")
def api_status():
    """status"""
    return jsonify(
        {
            "initialized": state["api"] is not None,
            "map_loaded": state["map"] is not None,
            "vehicles": len(state["vehicles"]) if state["vehicles"] else 0,
            "simulator_running": state["simulator"] is not None,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
