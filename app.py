import os
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import testing2 as t2
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_PATH = t2.MODEL_PATH

app = Flask(__name__)
app.secret_key = "replace-me"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)


def allowed_file(fname: str) -> bool:
    return Path(fname).suffix.lower() in ALLOWED_EXT


def save_latest_figure(out_path: Path) -> bool:
    try:
        managers = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
        if not managers:
            return False
        fig = managers[-1].canvas.figure
        fig.savefig(out_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return True
    except Exception:
        return False


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    f = request.files["image"]
    if f.filename == "":
        flash("Please choose an image.")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash("Unsupported file type. Use JPG/PNG/BMP/WEBP.")
        return redirect(url_for("index"))

    safe = secure_filename(f.filename)
    stem = Path(safe).stem[:40]
    ext = Path(safe).suffix.lower()
    unique_name = f"{stem}-{uuid.uuid4().hex[:8]}{ext}"
    img_path = UPLOAD_DIR / unique_name
    f.save(str(img_path))

    try:
        label, prob, explanation = t2.explain_image(str(img_path), model)

        gradcam_name = f"{Path(unique_name).stem}-gradcam.png"
        gradcam_path = UPLOAD_DIR / gradcam_name
        ok = save_latest_figure(gradcam_path)
        if not ok:
            fig = plt.figure(figsize=(6, 3))
            plt.axis("off")
            plt.text(0.5, 0.5, "Grad-CAM not available", ha="center", va="center")
            fig.savefig(gradcam_path, bbox_inches="tight", dpi=140)
            plt.close(fig)

        return render_template(
            "result.html",
            gradcam_image=url_for("static", filename=f"uploads/{gradcam_name}"),
            explanation=explanation
        )
    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
