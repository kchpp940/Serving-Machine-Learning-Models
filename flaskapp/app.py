from flask import Flask, render_template, request, jsonify
import utils
from model_bundle_loader import get_bundle, get_model, MODEL_BUNDLE_PATH
from car_pricing import build_model_info, ErrorMessages

app = Flask(__name__)


def _parse_form() -> tuple[dict, dict, dict]:
    bundle = get_bundle()
    fields = bundle.fields
    feature_order = bundle.feature_order

    form_data = request.form.to_dict()
    errors = {}
    model_features = {}

    names = form_data.get("names", "").strip()
    if not names:
        errors["names"] = "Car name cannot be empty."

    for field_name in feature_order:
        meta = fields[field_name]
        label = meta["label"]
        raw_value = form_data.get(field_name, "").strip()

        if not raw_value:
            errors[field_name] = ErrorMessages.FIELD_EMPTY.format(field=label)
            continue

        try:
            model_features[field_name] = bundle.encode_form_value(field_name, raw_value)
        except ValueError as e:
            errors[field_name] = str(e)

    return form_data, errors, model_features


def _template_context(extra=None):
    bundle = get_bundle()
    ctx = {
        "field_order": bundle.template_field_order,
        "fields": bundle.fields,
    }
    if extra:
        ctx.update(extra)
    return ctx


@app.route("/")
def home():
    return render_template("index.html", **_template_context())


@app.route("/model_info")
def model_info():
    model = get_model()
    info = build_model_info(model, MODEL_BUNDLE_PATH)
    return jsonify(info.to_dict())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        form_data, errors, model_features = _parse_form()

        if errors:
            return render_template(
                "index.html",
                **_template_context({"errors": errors, "form_data": form_data}),
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
            errors["_general"] = ErrorMessages.PREDICTION_FAILED.format(error=str(e))
            return render_template(
                "index.html",
                **_template_context({"errors": errors, "form_data": form_data}),
            )

    return render_template("index.html", **_template_context())


if __name__ == "__main__":
    app.run(debug=True)
