"""Flask application factory."""

from flask import Flask, jsonify

from .config import Config


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)

    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def root():
        return jsonify(
            service="Recipe Bot API",
            description="A RAG chatbot for Kenyan recipes.",
            endpoints={"health": "/api/health", "chat": "POST /api/chat"},
        )

    return app
