from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from .services import AgricultureService
from .crop_catalog import catalog_metadata


MAX_UPLOAD_MB = 8
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    service = AgricultureService()

    @app.after_request
    def add_cors_headers(response):
        response.headers.setdefault("Access-Control-Allow-Origin", "http://localhost:5173")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "crop_model": True,
            "soil_model": service.soil_available,
        })

    @app.get("/api/crops")
    def crops():
        return jsonify({"crops": service.available_crops(), "model_crops": service.model_crops()})

    @app.get("/api/crop-catalog")
    def crop_catalog():
        return jsonify(catalog_metadata(service.available_crops()))

    @app.get("/api/metrics")
    def metrics():
        return jsonify(service.metrics())

    @app.post("/api/analyze")
    def analyze():
        try:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                return jsonify({"error": "Request body must be valid JSON."}), 400
            values = {
                key: body.get(key)
                for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            }
            target = body.get("target_crop") or None
            return jsonify(service.analyze(values, target_crop=target))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            app.logger.exception("Analysis failed")
            return jsonify({"error": "Analysis failed. Check the server logs for details."}), 500

    @app.post("/api/analyze-with-image")
    def analyze_with_image():
        try:
            values = {
                key: request.form.get(key)
                for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            }
            target = request.form.get("target_crop") or None
            image = request.files.get("soil_image")
            if image is None or not image.filename:
                return jsonify({"error": "No soil image was uploaded."}), 400
            return jsonify(service.analyze(values, target_crop=target, soil_image=image))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            app.logger.exception("Image analysis failed")
            return jsonify({"error": "Image analysis failed. Check the server logs for details."}), 500

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        return jsonify({"error": f"Upload too large. Maximum size is {MAX_UPLOAD_MB} MB."}), 413

    # Serve a production React build from Flask when `npm run build` has been run.
    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def react_app(path: str):
        if not FRONTEND_DIST.exists():
            return jsonify({
                "message": "Flask backend is running.",
                "frontend": "Run 'npm install && npm run dev' inside frontend/ for development, or 'npm run build' for Flask-served production UI.",
            })
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file():
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    return app
