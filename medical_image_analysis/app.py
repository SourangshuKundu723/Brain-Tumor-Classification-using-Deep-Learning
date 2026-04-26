from flask import Flask, request, jsonify
import os
from lung_predict import predict_lung
from brain_predict import predict_brain
from flask import render_template
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")

# 🔹 Lung API
@app.route("/predict-lung", methods=["POST"])
def lung_api():
    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result, confidence = predict_lung(filepath)

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })


# 🔹 Brain API
@app.route("/predict-ct", methods=["POST"])   # keep name same as frontend
def brain_api():
    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result, confidence = predict_brain(filepath)

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)