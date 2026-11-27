import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

with open("KNN_Model.pkl", "rb") as f:
    model = pickle.load(f)

NUM_COLS = ["Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
            "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"]
CAT_COLS = ["Gender", "Exercise Habits", "Smoking", "Family Heart Disease", "Diabetes",
            "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
            "Alcohol Consumption", "Stress Level", "Sugar Consumption"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Form field names are the column names with spaces replaced by underscores
    row = {}
    for col in NUM_COLS:
        row[col] = float(request.form[col.replace(" ", "_")])
    for col in CAT_COLS:
        row[col] = request.form[col.replace(" ", "_")]

    features = pd.DataFrame([row])
    prob = model.predict_proba(features)[0][1]
    label = "Yes (heart disease)" if prob >= 0.5 else "No (no heart disease)"

    return render_template("index.html",
                           prediction_text=label,
                           probability=f"{prob * 100:.1f}",
                           is_yes=prob >= 0.5,
                           form=request.form)  # keeps the values the user typed


if __name__ == "__main__":
    app.run(debug=True)