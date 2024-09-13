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
    temperature: float = Field(..., ge=20.0, le=60.0)
    uv: int = Field(..., ge=0, le=15)
    uvA: int = Field(..., ge=0, le=7)
    LED: int = Field(..., ge=0, le=7)
    pump_a_volume: float = Field(..., ge=0.0, le=5.0)
    pump_a_speed: float = Field(..., ge=2000.0, le=4096.0)
    pump_b_volume: float = Field(..., ge=0.0, le=5.0)
    pump_b_speed: float = Field(..., ge=2000.0, le=4096.0)
    pump_c_volume: float = Field(..., ge=0.0, le=5.0)
    pump_c_speed: float = Field(..., ge=2000.0, le=4096.0)
    pump_d_volume: float = Field(..., ge=0.0, le=5.0)
    pump_d_speed: float = Field(..., ge=2000.0, le=4096.0)
    pump_e_volume: float = Field(..., ge=0.0, le=5.0)
    pump_e_speed: float = Field(..., ge=2000.0, le=4096.0)
    pump_f_volume: float = Field(..., ge=0.0, le=5.0)
    pump_f_speed: float = Field(..., ge=2000.0, le=4096.0)
    mixing_speed: float = Field(..., ge=2048.0, le=4096.0)
