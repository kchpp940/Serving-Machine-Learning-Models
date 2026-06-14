import joblib
import numpy as np


def predict_price(
    enginesize,
    curbweight,
    horsepower,
    highwaympg,
    carwidth,
    wheelbase,
    drivewheel,
    citympg,
    boreratio,
    cylindernumber,
):

    data = np.array(
        [
            [
                enginesize,
                curbweight,
                horsepower,
                highwaympg,
                carwidth,
                wheelbase,
                drivewheel,
                citympg,
                boreratio,
                cylindernumber,
            ]
        ]
    )

    model = joblib.load("./models/sklearn_gbr.pkl")
    predictions = model.predict(data)

    return predictions
