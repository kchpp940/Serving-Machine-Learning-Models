import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request
import utils

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES


app = Flask(__name__)

FIELD_LABELS = {
    "names": "Name of Car",
    "enginesize": "Engine Size",
    "curbweight": "Curb Weight",
    "horsepower": "Horse Power",
    "highwaympg": "Highway Miles Per Gallon",
    "carwidth": "Car Width",
    "wheelbase": "Wheel Base",
    "drivewheel": "Drive Wheel",
    "citympg": "City Miles Per Gallon",
    "boreratio": "Bore Ratio",
    "cylindernumber": "Number of Cylinders",
}

PREDICTION_FIELDS = list(FEATURE_ORDER)
CATEGORICAL_SET = set(CATEGORICAL_FEATURES)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        form_data = request.form.to_dict()
        errors = {}
        parsed = {}

        names = form_data.get("names", "").strip()
        if not names:
            errors["names"] = "Car name cannot be empty."

        for field in PREDICTION_FIELDS:
            raw_value = form_data.get(field, "").strip()
            if not raw_value:
                errors[field] = f"{FIELD_LABELS.get(field, field)} cannot be empty."
                continue
            if field in CATEGORICAL_SET:
                parsed[field] = raw_value
            else:
                try:
                    parsed[field] = float(raw_value)
                except (ValueError, TypeError):
                    errors[field] = (
                        f"{FIELD_LABELS.get(field, field)} must be a valid number (decimals allowed)."
                    )

        if errors:
            return render_template(
                "index.html", errors=errors, form_data=form_data
            )

        try:
            predicts = utils.predict_price(**parsed)
            value = str(predicts)[1:-1]
            return render_template(
                "result.html", result=f"The Price of the {names} is: {value}$"
            )
        except Exception as e:
            errors["_general"] = f"Prediction failed: {str(e)}"
            return render_template(
                "index.html", errors=errors, form_data=form_data
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
