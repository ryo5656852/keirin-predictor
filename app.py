from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")  # モデルは別途用意が必要

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form["data"]
        lines = data.strip().split("\n")
        features = [list(map(float, line.split(",")[1:])) for line in lines]
        names = [line.split(",")[0] for line in lines]
        X = np.array(features)
        preds = model.predict_proba(X)
        scores = preds[:, 1]
        results = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        top3 = results[:3]
        return render_template("index.html", results=top3)
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"
