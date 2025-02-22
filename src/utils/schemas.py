"""This module contains the schemas for the API."""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request schema for the API."""

    StoreCount: int
    ShelfCapacity: float
    PromoShelfCapacity: float
    IsPromo: bool
    ItemNumber: int
    CategoryCode: int
    GroupCode: int
    month: int
    weekday: int
    UnitSales_minus_7: float
    UnitSales_minus_14: float
    UnitSales_minus_21: float
