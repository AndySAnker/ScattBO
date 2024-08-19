from pydantic import BaseModel, Field


class PumpParameters(BaseModel):
    volume: float = Field(...)
    speed: float = Field(..., ge=2000.0, le=4096.0)
