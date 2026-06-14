from pydantic import BaseModel, Field


class CarPrediction(BaseModel):
    enginesize: float
    curbweight: float
    horsepower: float
    highwaympg: float
    carwidth: float
    wheelbase: float
    drivewheel: float
    citympg: float
    boreratio: float
    cylindernumber: float


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted car price")
    currency: str = Field(default="USD", description="Currency unit")
    model_name: str = Field(..., description="Name of the ML model used")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message describing the issue")
