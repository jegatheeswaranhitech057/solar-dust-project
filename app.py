"""
app.py
Flask backend for SolarSense Dust Detection System

Workflow
--------
1. User enters environmental data
2. ML model predicts expected energy
3. Compare with actual energy
4. Calculate energy loss
5. Classify dust level
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)
app.secret_key = "solar_dust_secret_2024"

USERNAME = "admin"
PASSWORD = "1234"

MODEL_PATH = os.path.join(app.root_path, "model", "energy_model.pkl")

# Model features (must match training data)
FEATURES = [
    "ambient_temp",
    "module_temp",
    "tilt_radiation",
    "peak_tilt_irradiation",
    "wind_speed",
    "plant_peak_power"
]

# Load trained ML model
MODEL = joblib.load(MODEL_PATH)

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("index"))

        else:
            flash("Invalid Username or Password")

    return render_template("login.html")


@app.route("/logout")
def logout():

    session.pop("user", None)
    return redirect(url_for("login"))


# -------------------------------
# Dust Classification
# -------------------------------
def classify_dust(loss):

    if loss < 5:
        return "Low", "No Cleaning Required", \
               "Panel performance is optimal."

    elif loss < 15:
        return "Medium", "Monitor Panel", \
               "Dust accumulation detected. Cleaning recommended soon."

    else:
        return "High", "Immediate Cleaning Required", \
               "Severe dust impact detected. Clean panels immediately."


# -------------------------------
# Predict Expected Energy
# -------------------------------
def predict_energy(features):

    X = np.array([features])
    predicted_energy = MODEL.predict(X)[0]

    return round(float(predicted_energy), 2)


# -------------------------------
# Calculate Energy Loss
# -------------------------------
def calculate_loss(predicted, actual):

    loss = ((predicted - actual) / predicted) * 100
    return round(loss, 2)


# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def index():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("index.html")

# -------------------------------
# Manual Prediction
# -------------------------------
@app.route("/predict_manual", methods=["POST"])
def predict_manual():

    try:
        ambient_temp = float(request.form["ambient_temp"])
        module_temp = float(request.form["module_temp"])
        tilt_radiation = float(request.form["tilt_radiation"])
        peak_tilt = float(request.form["peak_tilt_irradiation"])
        wind_speed = float(request.form["wind_speed"])
        plant_power = float(request.form["plant_peak_power"])
        actual_energy = float(request.form["actual_energy"])

    except ValueError:
        flash("Invalid input values", "danger")
        return redirect(url_for("index"))

    features = [
        ambient_temp,
        module_temp,
        tilt_radiation,
        peak_tilt,
        wind_speed,
        plant_power
    ]

    predicted_energy = predict_energy(features)

    loss = calculate_loss(predicted_energy, actual_energy)

    level, action, detail = classify_dust(loss)

    result = {
        "level": level,
        "confidence": round(100 - loss, 2),
        "predicted_energy": predicted_energy,
        "actual_energy": actual_energy,
        "loss": loss,
        "action": action,
        "detail": detail,
        "source": "Manual Input",
        "rows": None
    }

    email = request.form.get("email")

    if email:
        send_email(email, result)

    return render_template("result.html", result=result)


# -------------------------------
# CSV Prediction
# -------------------------------
@app.route("/predict_csv", methods=["POST"])
def predict_csv():

    if "csv_file" not in request.files:
        flash("No CSV uploaded", "danger")
        return redirect(url_for("index"))

    file = request.files["csv_file"]

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except:
        flash("Error reading CSV file", "danger")
        return redirect(url_for("index"))

    required_columns = FEATURES + ["actual_energy"]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        flash(f"Missing columns: {missing}", "danger")
        return redirect(url_for("index"))

    rows = []

    for i, row in df.iterrows():

        features = [
            row["ambient_temp"],
            row["module_temp"],
            row["tilt_radiation"],
            row["peak_tilt_irradiation"],
            row["wind_speed"],
            row["plant_peak_power"]
        ]

        predicted_energy = predict_energy(features)

        actual_energy = row["actual_energy"]

        loss = calculate_loss(predicted_energy, actual_energy)

        level, action, detail = classify_dust(loss)

        rows.append({
            "row": i + 1,
            "predicted_energy": predicted_energy,
            "actual_energy": actual_energy,
            "loss": loss,
            "level": level,
            "confidence": round(100 - loss, 2),
            "action": action
        })

    worst = max(rows, key=lambda x: x["loss"])

    result = {
        "level": worst["level"],
        "confidence": worst["confidence"],
        "action": worst["action"],
        "detail": "Highest dust impact detected in dataset",
        "source": f"CSV Upload ({len(rows)} rows)",
        "rows": rows
    }

    return render_template("result.html", result=result)


# -------------------------------
# Email Notification
# -------------------------------
def send_email(receiver, result):

    sender = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")

    subject = "Solar Panel Dust Prediction Result"

    body = f"""
Solar Dust Prediction Result

Predicted Energy: {result['predicted_energy']} kWh
Actual Energy: {result['actual_energy']} kWh
Energy Loss: {result['loss']} %

Dust Level: {result['level']}
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)