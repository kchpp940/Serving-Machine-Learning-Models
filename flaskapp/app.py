import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request

import utils

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import find_model_path, SchemaMismatchError


app = Flask(__name__)

_model: CarPriceModel = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        local_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = find_model_path(local_dir=local_dir)
        _model = CarPriceModel.from_joblib(model_path)
        from car_pricing.feature_schema import FEATURE_ORDER
        _model.validate_service(list(FEATURE_ORDER))
    return _model


EXTRA_FORM_FIELDS = ["names"]


def form_fields():
    return get_model().form_fields_metadata(extra_fields=EXTRA_FORM_FIELDS)


@app.route("/")
def home():
    fields_meta = form_fields()
    return render_template(
        "index.html",
        fields=fields_meta,
        errors={},
        form_data={},
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        model = get_model()
        form_data = request.form.to_dict()
        errors = {}
        parsed = {}
        names = form_data.get("names", "").strip()
        if not names:
            errors["names"] = model.field_error_message("names", "required") or "Car name cannot be empty."

        for field in model.feature_order:
            raw_value = form_data.get(field, "").strip()
            if not raw_value:
                errors[field] = model.field_error_message(field, "required")
                continue
            if field in model.categorical_features:
                allowed = set(model.field_allowed_values(field))
                if raw_value not in allowed:
                    errors[field] = model.field_error_message(field, "invalid")
                    continue
                parsed[field] = raw_value
            else:
                try:
                    parsed[field] = float(raw_value)
                except (ValueError, TypeError):
                    errors[field] = model.field_error_message(field, "type")

        if errors:
            return render_template(
                "index.html",
                fields=form_fields(),
                errors=errors,
                form_data=form_data,
            )

        try:
            predicts = utils.predict_price(**parsed)
            value = str(predicts)[1:-1]
            return render_template(
                "result.html",
                result=f"The Price of the {names} is: {value}$",
            )
        except SchemaMismatchError as e:
            errors["_general"] = f"Schema 不一致: {str(e)}"
        except Exception as e:
            errors["_general"] = f"Prediction failed: {str(e)}"

        return render_template(
            "index.html",
            fields=form_fields(),
            errors=errors,
            form_data=form_data,
        )

    fields_meta = form_fields()
    return render_template(
        "index.html",
        fields=fields_meta,
        errors={},
        form_data={},
    )


if __name__ == "__main__":
    app.run(debug=True)
