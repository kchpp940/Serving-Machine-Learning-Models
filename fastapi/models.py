from typing import Optional
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
    class Config:
        protected_namespaces = ()

    prediction: float = Field(..., description="Predicted car price")
    currency: str = Field(default="USD", description="Currency unit")
    model_name: str = Field(..., description="Name of the ML model used")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message describing the issue")


class HealthResponse(BaseModel):
    class Config:
        protected_namespaces = ()

    status: str = Field(..., description="Service health status: ok or degraded")
    model_available: bool = Field(..., description="Whether the ML model is loaded and ready")
    model_error: Optional[str] = Field(default=None, description="Error message if model failed to load")
