from typing import Literal

from pydantic import BaseModel, Field

from ScattBO.parameters.pumps_parameters import PumpParameters


class LightParameters(BaseModel):
    type_: Literal["uvA", "uvC", "visible"]
    amount: int = Field(..., ge=0, le=15)
    pulsing_frequency: float


class MixingParameters(BaseModel):
    mixing_speed: float = Field(..., ge=2048.0, le=4096.0)
    mixing_time: float


class ExperimentParameters(BaseModel):
    pumps: list[PumpParameters]
    volume: float
    metadata_for_database: str
    mixing_pattern: MixingParameters
    light_patterns: list[LightParameters]
    temperature: float = Field(..., ge=20.0, le=70.0)
