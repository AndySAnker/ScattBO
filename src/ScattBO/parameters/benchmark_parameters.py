from typing import Literal

from pydantic import BaseModel, Field

from ScattBO.parameters.pumps_parameters import PumpsParameters


class BenchmarkParameters(BaseModel):
    pH: float
    pressure: float
    solvent: Literal["Ethanol", "Methanol", "Water", "Others"]


class SmallBenchmarkParameters(BenchmarkParameters):
    pH: float = Field(..., ge=2.0, le=12.0)
    pressure: float = Field(..., ge=15.0, le=80.0)
    solvent: Literal["Ethanol", "Methanol"]


class LargeBenchmarkParameters(BenchmarkParameters):
    pH: float = Field(..., ge=0.0, le=14.0)
    pressure: float = Field(..., ge=0.0, le=100.0)
    solvent: Literal["Ethanol", "Methanol", "Water", "Others"]


class RoboticBenchmarkParameters(BaseModel):
    temperature: float = Field(..., ge=20.0, le=70.0)
    uv: int = Field(..., ge=0, le=15)
    uvA: int = Field(..., ge=0, le=7)
    LED: int = Field(..., ge=0, le=7)
    pumps: list[PumpsParameters] = Field(..., min_length=6, max_length=6)
    mixing_speed: float = Field(..., ge=2048.0, le=4096.0)
    atom: str = Field(default="Au")
