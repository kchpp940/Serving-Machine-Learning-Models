MODEL_FEATURE_ORDER = [
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

FIELDS = {
    "names": {
        "type": "text",
        "label": "Name of Car",
        "placeholder": "Enter Name of Car (e.g. Toyota Camry)",
        "is_model_feature": False,
    },
    "enginesize": {
        "type": "numeric",
        "label": "Engine Size",
        "placeholder": "Engine displacement in cubic inches (e.g. 130)",
        "is_model_feature": True,
    },
    "curbweight": {
        "type": "numeric",
        "label": "Curb Weight",
        "placeholder": "Weight in pounds (e.g. 2548)",
        "is_model_feature": True,
    },
    "horsepower": {
        "type": "numeric",
        "label": "Horse Power",
        "placeholder": "Horsepower output (e.g. 111)",
        "is_model_feature": True,
    },
    "highwaympg": {
        "type": "numeric",
        "label": "Highway Miles Per Gallon",
        "placeholder": "Highway fuel efficiency (e.g. 27)",
        "is_model_feature": True,
    },
    "carwidth": {
        "type": "numeric",
        "label": "Car Width",
        "placeholder": "Width in inches (e.g. 64.1)",
        "is_model_feature": True,
    },
    "wheelbase": {
        "type": "numeric",
        "label": "Wheel Base",
        "placeholder": "Distance between front and rear axles in inches (e.g. 88.6)",
        "is_model_feature": True,
    },
    "drivewheel": {
        "type": "categorical",
        "label": "Drive Wheel",
        "placeholder": "Select drive wheel configuration",
        "is_model_feature": True,
        "options": [
            {"display": "Front Wheel Drive (FWD)", "form_value": "fwd", "model_code": 1},
            {"display": "Rear Wheel Drive (RWD)", "form_value": "rwd", "model_code": 2},
            {"display": "Four Wheel Drive (4WD)", "form_value": "4wd", "model_code": 0},
        ],
        "form_to_model": {"4wd": 0, "fwd": 1, "rwd": 2},
    },
    "citympg": {
        "type": "numeric",
        "label": "City Miles Per Gallon",
        "placeholder": "City fuel efficiency (e.g. 21)",
        "is_model_feature": True,
    },
    "boreratio": {
        "type": "numeric",
        "label": "Bore Ratio",
        "placeholder": "Engine bore ratio (e.g. 3.47)",
        "is_model_feature": True,
    },
    "cylindernumber": {
        "type": "categorical",
        "label": "Number of Cylinders",
        "placeholder": "Select number of engine cylinders",
        "is_model_feature": True,
        "options": [
            {"display": "2 cylinders", "form_value": "two", "model_code": 6},
            {"display": "3 cylinders", "form_value": "three", "model_code": 4},
            {"display": "4 cylinders", "form_value": "four", "model_code": 2},
            {"display": "5 cylinders", "form_value": "five", "model_code": 1},
            {"display": "6 cylinders", "form_value": "six", "model_code": 3},
            {"display": "8 cylinders", "form_value": "eight", "model_code": 0},
            {"display": "12 cylinders", "form_value": "twelve", "model_code": 5},
        ],
        "form_to_model": {
            "eight": 0,
            "five": 1,
            "four": 2,
            "six": 3,
            "three": 4,
            "twelve": 5,
            "two": 6,
        },
    },
}

MODEL_NUMERIC_FEATURES = [
    f for f in MODEL_FEATURE_ORDER if FIELDS[f]["type"] == "numeric"
]

MODEL_CATEGORICAL_FEATURES = [
    f for f in MODEL_FEATURE_ORDER if FIELDS[f]["type"] == "categorical"
]

TEMPLATE_FIELD_ORDER = [
    "names",
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


def form_value_to_model_code(field_name: str, form_value: str) -> float:
    field = FIELDS[field_name]
    if field["type"] == "numeric":
        return float(form_value)
    if field["type"] == "categorical":
        return float(field["form_to_model"][form_value])
    raise ValueError(f"Unsupported field type: {field['type']}")


def get_template_context():
    return {
        "field_order": TEMPLATE_FIELD_ORDER,
        "fields": FIELDS,
    }
