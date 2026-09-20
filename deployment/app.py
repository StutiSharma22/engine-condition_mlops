
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("best_engine_condition_model.joblib")

@app.route("/")
def home():
    return "Engine Condition Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)

    return jsonify({
        "prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860
    )
