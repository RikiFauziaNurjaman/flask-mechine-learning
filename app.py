from flask import Flask, jsonify, request
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_model = joblib.load("models/model_diabetes.pkl")

load_scaler = joblib.load("models/standart_scaler.pkl")

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@app.route("/")
def index():
    return jsonify(
        {"meta": {"status": "success", "message": "Welcome to Api"}, "data": None}
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    x_input = pd.DataFrame([data], columns=columns)

    x_input_scaled = load_scaler.transform(x_input)

    prediction = load_model.predict(x_input_scaled)

    return jsonify(
        {
            "meta": {"status": "success", "message": "Prediction"},
            "data": prediction.tolist()[0],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
