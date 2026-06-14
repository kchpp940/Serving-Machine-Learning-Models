from __future__ import annotations

from pydantic import BaseModel, Field, validator


class CarPrediction(BaseModel):
    """FastAPI 请求体。

    - 数值字段允许 int/float；
    - 分类字段（drivewheel / cylindernumber）优先接受人类可读字符串，
      同时保留对旧客户端传入数字编码的向后兼容。
    """

    enginesize: float = Field(description="Engine displacement in cubic inches, e.g. 130")
    curbweight: float = Field(description="Curb weight in pounds, e.g. 2548")
    horsepower: float = Field(description="Horsepower output, e.g. 111")
    highwaympg: float = Field(description="Highway miles per gallon, e.g. 27")
    carwidth: float = Field(description="Car width in inches, e.g. 64.1")
    wheelbase: float = Field(description="Wheel base in inches, e.g. 88.6")
    drivewheel: str | int | float = Field(
        description="Drive wheel: '4wd' / 'fwd' / 'rwd' (preferred) or legacy numeric code 0/1/2"
    )
    citympg: float = Field(description="City miles per gallon, e.g. 21")
    boreratio: float = Field(description="Engine bore ratio, e.g. 3.47")
    cylindernumber: str | int | float = Field(
        description="Cylinder count word: 'two' / 'three' / 'four' / 'five' / 'six' / 'eight' / 'twelve' "
                    "(preferred) or legacy numeric code"
    )

    @validator("drivewheel", "cylindernumber", pre=True, always=True)
    def _coerce_categorical(cls, v):
        if v is None:
            raise ValueError("cannot be null")
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, (int, float)):
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return v
        return str(v).strip()
