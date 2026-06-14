from flask import Flask, render_template, request
import utils
from schema import (
    FIELDS,
    MODEL_FEATURE_ORDER,
    TEMPLATE_FIELD_ORDER,
    form_value_to_model_code,
)

app = Flask(__name__)


def _get_label(field_name: str) -> str:
    return FIELDS[field_name]["label"]


def _parse_form() -> tuple[dict, dict, dict]:
    form_data = request.form.to_dict()
    errors = {}
    model_features = {}

    names = form_data.get("names", "").strip()
    if not names:
        errors["names"] = "Car name cannot be empty."

    for field_name in MODEL_FEATURE_ORDER:
        field = FIELDS[field_name]
        raw_value = form_data.get(field_name, "").strip()

        if not raw_value:
            errors[field_name] = f"{_get_label(field_name)} cannot be empty."
            continue

        if field["type"] == "numeric":
            try:
                model_features[field_name] = float(raw_value)
            except (ValueError, TypeError):
                errors[field_name] = (
                    f"{_get_label(field_name)} must be a valid number (decimals allowed)."
                )

        elif field["type"] == "categorical":
            valid_values = set(opt["form_value"] for opt in field["options"])
            if raw_value not in valid_values:
                valid_labels = ", ".join(
                    opt["display"] for opt in field["options"]
                )
                errors[field_name] = (
                    f"{_get_label(field_name)} must be one of: {valid_labels}."
                )
            else:
                model_features[field_name] = form_value_to_model_code(
                    field_name, raw_value
                )

    return form_data, errors, model_features


@app.route("/")
def home():
    return render_template(
        "index.html",
        field_order=TEMPLATE_FIELD_ORDER,
        fields=FIELDS,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        form_data, errors, model_features = _parse_form()

        if errors:
            return render_template(
                "index.html",
                errors=errors,
                form_data=form_data,
                field_order=TEMPLATE_FIELD_ORDER,
                fields=FIELDS,
            )

        try:
            predictions = utils.predict_price(model_features)
            value = str(predictions)[1:-1]
            car_name = form_data.get("names", "").strip()
            return render_template(
                "result.html",
                result=f"The Price of the {car_name} is: {value}$",
            )
        except Exception as e:
            errors["_general"] = f"Prediction failed: {str(e)}"
            return render_template(
                "index.html",
                errors=errors,
                form_data=form_data,
                field_order=TEMPLATE_FIELD_ORDER,
                fields=FIELDS,
            )

    return render_template(
        "index.html",
        field_order=TEMPLATE_FIELD_ORDER,
        fields=FIELDS,
    )


if __name__ == "__main__":
    app.run(debug=True)
