import os
import io
import json
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Optional: .env support during local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========================
# Flask App & Basic Config
# =========================
app = Flask(__name__)

# (Opsional) batasi ukuran upload, mis. 15MB
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", "15")) * 1024 * 1024

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model/model_CNN.h5")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # untuk output biner jika dibutuhkan

# Default class names (index -> label). Bisa override via env CLASS_NAMES (JSON list).
DEFAULT_CLASS_NAMES = ["Human", "AI"]
try:
    CLASS_NAMES = json.loads(os.getenv("CLASS_NAMES", ""))
    if not isinstance(CLASS_NAMES, list) or len(CLASS_NAMES) == 0:
        CLASS_NAMES = DEFAULT_CLASS_NAMES
except Exception:
    CLASS_NAMES = DEFAULT_CLASS_NAMES

# -----------------------------
# Load model
# -----------------------------
import tensorflow as tf
from tensorflow.keras.models import load_model

# Jika ada custom layer, definisikan di custom_objects
# custom_objects = {"Swish": tf.nn.swish, ...}
# model = load_model(MODEL_PATH, custom_objects=custom_objects)
model = load_model(MODEL_PATH)

# Coba infer input shape, mis. (None, H, W, C)
input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]  # handle multi-input
# fallback ukuran
if len(input_shape) == 4:
    _, H, W, C = input_shape
elif len(input_shape) == 3:
    H, W, C = input_shape
else:
    H, W, C = 224, 224, 3

# ================
# Helper Functions
# ================
def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("RGB")
    # PIL pakainya (width, height)
    img = img.resize((W, H))
    arr = np.array(img).astype("float32") / 255.0  # samakan dg training
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr

def preprocess_image_with_meta(file_storage):
    """
    Membaca file, mengembalikan (numpy_input, meta_dict).
    NOTE: file_storage.read() menghabiskan stream; aman karena kita tidak pakai ulang objek file_storage.
    """
    raw = file_storage.read()
    size_kb = len(raw) / 1024.0
    bio = io.BytesIO(raw)

    img = Image.open(bio).convert("RGB")
    orig_w, orig_h = img.size  # PIL: (width, height)

    img = img.resize((W, H))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    meta = {
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "file_kb": round(size_kb, 1),
        "input_shape": [H, W, C],
        "threshold": float(THRESHOLD),
        "model_name": getattr(model, "name", "model"),
    }
    return arr, meta

def postprocess_logits(logits):
    """
    Normalisasi output model ke struktur standar:
    - 1 unit (sigmoid): mapping ke [Human, AI]
    - >=2 unit: asumsikan softmax (atau disoftmax-kan kalau belum)
    """
    logits = np.array(logits)
    if logits.ndim == 2 and logits.shape[1] == 1:
        prob_ai = float(logits[0, 0])  # asumsi label 1 = "AI"
        probs = [1.0 - prob_ai, prob_ai]  # [Human, AI]
    elif logits.ndim == 2 and logits.shape[1] >= 2:
        row = logits[0]
        if not np.isclose(np.sum(row), 1.0):
            e = np.exp(row - np.max(row))
            row = e / np.sum(e)
        probs = row.tolist()
    else:
        prob_ai = float(np.ravel(logits)[0])
        probs = [1.0 - prob_ai, prob_ai]

    names = CLASS_NAMES if len(CLASS_NAMES) == len(probs) else DEFAULT_CLASS_NAMES[:len(probs)]
    top_idx = int(np.argmax(probs))
    return {
        "classes": [{"label": names[i], "prob": float(probs[i])} for i in range(len(probs))],
        "top": {"label": names[top_idx], "prob": float(probs[top_idx])}
    }

# =========
# Routes
# =========
@app.get("/healthz")
def healthz():
    """Endpoint untuk health check di HF Spaces"""
    try:
        # ping super cepat agar siap dipakai autoscaler
        _ = getattr(model, "name", "model")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    # Jika kamu punya templates/index.html di dalam container, ini akan dirender.
    # Kalau tidak ada, fallback ke teks sederhana agar tidak 500.
    try:
        return render_template("index.html")
    except Exception:
        return (
            "AI Image Detector (Flask). "
            "Gunakan POST /predict (form field 'image') atau /api/predict (form field 'image').",
            200,
        )

@app.post("/predict")
def predict_form():
    if "image" not in request.files:
        return render_template("index.html", error="Tidak ada field file 'image'.")
    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="Tidak ada file yang dipilih.")
    filename = secure_filename(file.filename)
    try:
        x, meta = preprocess_image_with_meta(file)

        t0 = time.perf_counter()
        logits = model.predict(x)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        result = postprocess_logits(logits)

        # ambil AI & Human prob dari result
        ai_prob = None
        for c in result["classes"]:
            if str(c["label"]).lower() == "ai":
                ai_prob = float(c["prob"])
                break
        if ai_prob is None:
            ai_prob = float(result["top"]["prob"]) if str(result["top"]["label"]).lower() == "ai" else 1.0 - float(result["top"]["prob"])
        human_prob = 1.0 - ai_prob
        margin = abs(ai_prob - human_prob)

        result["meta"] = {
            **meta,
            "ai_prob": ai_prob,
            "human_prob": human_prob,
            "margin": round(margin, 4),
            "inference_ms": round(infer_ms, 1),
            "filename": filename,
        }
        return render_template("index.html", result=result, filename=filename)
    except Exception as e:
        # Jika template tidak ada, fallback JSON agar tidak blank
        try:
            return render_template("index.html", error=f"Error: {str(e)}")
        except Exception:
            return jsonify({"success": False, "error": str(e)}), 500

@app.post("/api/predict")
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file field"}), 400
    file = request.files["image"]
    try:
        x, meta = preprocess_image_with_meta(file)

        t0 = time.perf_counter()
        logits = model.predict(x)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        result = postprocess_logits(logits)

        ai_prob = None
        for c in result["classes"]:
            if str(c["label"]).lower() == "ai":
                ai_prob = float(c["prob"])
                break
        if ai_prob is None:
            ai_prob = float(result["top"]["prob"]) if str(result["top"]["label"]).lower() == "ai" else 1.0 - float(result["top"]["prob"])
        human_prob = 1.0 - ai_prob
        margin = abs(ai_prob - human_prob)

        result["meta"] = {
            **meta,
            "ai_prob": ai_prob,
            "human_prob": human_prob,
            "margin": round(margin, 4),
            "inference_ms": round(infer_ms, 1),
        }
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# Local Dev Entrypoint Only
# =========================
if __name__ == "__main__":
    # HF Spaces Docker mengharapkan service di port 7860
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")), debug=True)
