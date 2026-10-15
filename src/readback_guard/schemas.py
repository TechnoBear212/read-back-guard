from typing import Literal

from pydantic import BaseModel

SCORED_TYPES = ("level", "heading", "speed", "frequency", "squawk", "runway")


class Item(BaseModel):
    type: Literal["level", "heading", "speed", "frequency", "squawk", "qnh", "runway"]
    value: str
    unit: Literal["FL", "ft"] | None = None
    qualifier: Literal["climb", "descend", "left", "right", "increase", "reduce"] | None = None


class Clearance(BaseModel):
    callsign: str | None
    items: list[Item]


class ItemResult(BaseModel):
    type: str
    atco_value: str
    pilot_value: str | None
    result: Literal["OK", "WRONG", "MISSING"]


class Verdict(BaseModel):
    verdict: Literal["CORRECT", "WRONG_VALUE", "MISSING_ITEM", "WRONG_CALLSIGN", "SAY_AGAIN"]
    reason: str
    items: list[ItemResult] = []
