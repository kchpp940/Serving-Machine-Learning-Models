from flask import Flask, render_template, request, jsonify
import utils

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

NUMERIC_FIELDS = [
    "enginesize",
    "curbweight",
    "horsepower",
    "highwaympg",
    "carwidth",
    "wheelbase",
    "drivewheel",
    "citympg",
    "boreratio",
    "cylindernumber",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/schema")
def api_schema():
    try:
        model = utils._get_model()
        categorical_options = {}
        for f in model.categorical_features:
            categorical_options[f] = model.categorical_options(f)
        return jsonify({
            "feature_order": model.feature_order,
            "numeric_features": model.numeric_features,
            "categorical_features": model.categorical_features,
            "target_column": model.target_column,
            "categorical_options": categorical_options,
        })
    except Exception as e:
        return jsonify({"detail": f"Failed to load schema: {str(e)}"}), 500


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        form_data = request.form.to_dict()
        errors = {}
        parsed = {}

        names = form_data.get("names", "").strip()
        if not names:
            errors["names"] = "Car name cannot be empty."

        for field in NUMERIC_FIELDS:
            raw_value = form_data.get(field, "").strip()
            if not raw_value:
                errors[field] = f"{FIELD_LABELS[field]} cannot be empty."
                continue
            try:
                parsed[field] = float(raw_value)
            except (ValueError, TypeError):
                errors[field] = (
                    f"{FIELD_LABELS[field]} must be a valid number (decimals allowed)."
                )

        if errors:
            return render_template(
                "index.html", errors=errors, form_data=form_data
            )

        try:
            predicts = utils.predict_price(
                parsed["enginesize"],
                parsed["curbweight"],
                parsed["horsepower"],
                parsed["highwaympg"],
                parsed["carwidth"],
                parsed["wheelbase"],
                parsed["drivewheel"],
                parsed["citympg"],
                parsed["boreratio"],
                parsed["cylindernumber"],
            )
            value = float(predicts[0])
            return render_template(
                "result.html", result=f"The Price of the {names} is: {value:.2f}$"
            )
        except Exception as e:
            errors["_general"] = f"Prediction failed: {str(e)}"
            return render_template(
                "index.html", errors=errors, form_data=form_data
            )

    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"detail": "Request body must be valid JSON."}), 400

    required_fields = [
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "drivewheel", "citympg",
        "boreratio", "cylindernumber",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"detail": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        parsed = {}
        for f in required_fields:
            try:
                parsed[f] = float(data[f]) if f not in ("drivewheel", "cylindernumber") else str(data[f])
            except (ValueError, TypeError):
                return jsonify({"detail": f"Invalid value for field '{f}'."}), 400

        predicts = utils.predict_price(
            parsed["enginesize"],
            parsed["curbweight"],
            parsed["horsepower"],
            parsed["highwaympg"],
            parsed["carwidth"],
            parsed["wheelbase"],
            parsed["drivewheel"],
            parsed["citympg"],
            parsed["boreratio"],
            parsed["cylindernumber"],
        )
        value = float(predicts[0])
        return jsonify({"prediction": value, "status": "ok"})
    except Exception as e:
        return jsonify({"detail": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
