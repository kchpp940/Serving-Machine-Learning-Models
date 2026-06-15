from flask import Flask, render_template, request, jsonify
import utils

from car_pricing.feature_schema import FeatureSchema

app = Flask(__name__)

_default_schema = FeatureSchema.default()


def _field_label(field_name: str) -> str:
    return _default_schema.field_display_name(field_name)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/schema")
def api_schema():
    try:
        model = utils._get_model()
        return jsonify(model.to_schema_dict(include_encoders=True))
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

        schema = _default_schema
        for field in schema.feature_order:
            raw_value = form_data.get(field, "").strip()
            if not raw_value:
                errors[field] = f"{_field_label(field)} cannot be empty."
                continue
            try:
                if field in schema.numeric_features:
                    parsed[field] = float(raw_value)
                else:
                    parsed[field] = str(raw_value)
            except (ValueError, TypeError):
                errors[field] = (
                    f"{_field_label(field)} must be a valid number (decimals allowed)."
                )

        if errors:
            return render_template(
                "index.html", errors=errors, form_data=form_data
            )

        try:
            predicts = utils.predict_price(parsed)
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

    schema = _default_schema
    required_fields = schema.feature_order
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"detail": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        parsed = {}
        for f in required_fields:
            try:
                if f in schema.numeric_features:
                    parsed[f] = float(data[f])
                else:
                    parsed[f] = str(data[f])
            except (ValueError, TypeError):
                return jsonify({"detail": f"Invalid value for field '{f}'."}), 400

        predicts = utils.predict_price(parsed)
        value = float(predicts[0])
        return jsonify({"prediction": value, "status": "ok"})
    except Exception as e:
        return jsonify({"detail": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
