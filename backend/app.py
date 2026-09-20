import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from analyzer import analyze
from resume_data import ResumeError
from scoring import get_encoder

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin during dev) to call this API

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Load the sentence-transformer model once at startup, not on every request.
# First request/startup downloads ~90MB if not cached; after that it's local.
_encoder = None
try:
    _encoder = get_encoder()
except Exception as e:
    print(f"WARNING: semantic model failed to load, semantic scoring disabled: {e}")


@app.route("/api/analyze", methods=["POST"])
def analyze_resume():
    jd_text = request.form.get("jd_text", "").strip()
    if not jd_text or len(jd_text) < 30:
        return jsonify({"error": "Please paste a longer job description (30+ characters)."}), 400

    if "resume" not in request.files:
        return jsonify({"error": "No resume file was uploaded."}), 400
    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No resume file was selected."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF and DOCX files are supported."}), 400

    # Save to a temp file because our parser (pdfplumber/python-docx) needs a path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = analyze(tmp_path, jd_text, encoder=_encoder, use_semantic=_encoder is not None)
        return jsonify(result)
    except ResumeError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Don't leak stack traces to the frontend; log server-side instead.
        app.logger.exception("analyze failed")
        return jsonify({"error": "Something went wrong while analyzing the resume."}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "semantic_model_loaded": _encoder is not None})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
