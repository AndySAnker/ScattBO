from pydantic import BaseModel, Field


class PumpsParameters(BaseModel):
    volume: float = Field(...)
    speed: float = Field(..., ge=2000.0, le=4096.0)
