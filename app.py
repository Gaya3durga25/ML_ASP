import os
import sys
from flask import Flask, request, render_template, jsonify

from src.exception import CustomException
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Pull form data
        data = CustomData(
            Defect_of_vehicle           = request.form.get("Defect_of_vehicle"),
            Age_band_of_driver          = request.form.get("Age_band_of_driver"),
            Road_surface_type           = request.form.get("Road_surface_type"),
            Vehicle_driver_relation     = request.form.get("Vehicle_driver_relation"),
            Weather_conditions          = request.form.get("Weather_conditions"),
            Types_of_Junction           = request.form.get("Types_of_Junction"),
            Light_conditions            = request.form.get("Light_conditions"),
            Day_of_week                 = request.form.get("Day_of_week"),
            Number_of_casualties        = int(request.form.get("Number_of_casualties")),
            Number_of_vehicles_involved = int(request.form.get("Number_of_vehicles_involved")),
        )

        df       = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        result   = pipeline.predict(df)

        severity = result[0]

        # Severity → color/icon mapping for UI
        severity_meta = {
            "Slight":  {"color": "slight",  "icon": "⚠️"},
            "Serious": {"color": "serious", "icon": "🔶"},
            "Fatal":   {"color": "fatal",   "icon": "🔴"},
        }
        meta = severity_meta.get(severity, {"color": "slight", "icon": "⚠️"})

        return render_template(
            "index.html",
            prediction=severity,
            color=meta["color"],
            icon=meta["icon"],
            submitted=True,
            form_data=request.form,
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback to 5000 locally
    app.run(host="0.0.0.0", port=port)