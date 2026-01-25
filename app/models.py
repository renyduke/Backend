from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

# ============================================================================
# Request Models
# ============================================================================

class ForecastRequest(BaseModel):
    commodity: str = Field(..., description="Name of the commodity to forecast")
    data_type: str = Field(..., description="Type of data: 'volume' or 'price'")
    weeks_ahead: int = Field(default=4, ge=1, le=12, description="Number of weeks to forecast (1-12)")
    
    @validator('data_type')
    def validate_data_type(cls, v):
        if v not in ['volume', 'price']:
            raise ValueError("data_type must be either 'volume' or 'price'")
        return v
    
    @validator('commodity')
    def validate_commodity(cls, v):
        if not v or not v.strip():
            raise ValueError("commodity cannot be empty")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "commodity": "Cabbage",
                "data_type": "volume",
                "weeks_ahead": 4
            }
        }


class WeeklyDataPoint(BaseModel):
    year: int = Field(..., description="Year")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    week: int = Field(..., ge=1, le=5, description="Week of the month (1-5)")
    week_label: str = Field(..., description="Human-readable week label")
    period: str = Field(..., description="Sortable period key (YYYY-MM-W)")
    value: float = Field(..., description="Value for the period")

    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "month": 12,
                "week": 4,
                "week_label": "Week 4 (Days 22-28)",
                "period": "2024-12-04",
                "value": 1250.50
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class ForecastResponse(BaseModel):
    commodity: str = Field(..., description="Commodity name")
    data_type: str = Field(..., description="Type of data forecasted")
    historical_data: List[WeeklyDataPoint] = Field(..., description="Historical weekly data")
    forecast_data: List[WeeklyDataPoint] = Field(..., description="Forecasted weekly data")
    metrics: dict = Field(..., description="Model performance metrics")

    class Config:
        schema_extra = {
            "example": {
                "commodity": "Cabbage",
                "data_type": "volume",
                "historical_data": [
                    {
                        "year": 2024,
                        "month": 11,
                        "week": 1,
                        "week_label": "Week 1 (Days 1-7)",
                        "period": "2024-11-01",
                        "value": 1200.0
                    }
                ],
                "forecast_data": [
                    {
                        "year": 2024,
                        "month": 12,
                        "week": 1,
                        "week_label": "Week 1 (Days 1-7)",
                        "period": "2024-12-01",
                        "value": 1250.0
                    }
                ],
                "metrics": {
                    "mse": 125.5,
                    "rmse": 11.2,
                    "mae": 8.5,
                    "mape": 5.2,
                    "accuracy": 94.8
                }
            }
        }


class DashboardData(BaseModel):
    volume_data: List[dict] = Field(..., description="Weekly volume data for all commodities")
    price_data: List[dict] = Field(..., description="Weekly price data for all commodities")
    commodities: List[str] = Field(..., description="List of available commodities")
    period_range: dict = Field(..., description="Range of available periods")

    class Config:
        schema_extra = {
            "example": {
                "volume_data": [
                    {
                        "year": 2024,
                        "month": 12,
                        "week": 1,
                        "week_label": "Week 1 (Days 1-7)",
                        "period": "2024-12-01",
                        "commodity": "Cabbage",
                        "volume": 1200.0
                    }
                ],
                "price_data": [
                    {
                        "year": 2024,
                        "month": 12,
                        "week": 1,
                        "week_label": "Week 1 (Days 1-7)",
                        "period": "2024-12-01",
                        "commodity": "Cabbage",
                        "lowest_price": 25.0,
                        "highest_price": 35.0,
                        "average_price": 30.0
                    }
                ],
                "commodities": ["Cabbage", "Carrots", "Tomato"],
                "period_range": {
                    "start": "2024-01-01",
                    "end": "2024-12-04"
                }
            }
        }


class CommoditiesResponse(BaseModel):
    commodities: List[str] = Field(..., description="List of available commodities")

    class Config:
        schema_extra = {
            "example": {
                "commodities": [
                    "Beans", "Broccoli", "Cabbage", "Camote", "Carrots",
                    "Cauliflower", "Cucumber", "Eggplant", "Ginger",
                    "Green Onion", "Lettuce", "Pechay", "Pepper",
                    "Potato", "Radish", "Sayote", "Squash", "Tomato"
                ]
            }
        }


class StatisticsResponse(BaseModel):
    commodity: str = Field(..., description="Commodity name")
    data_type: str = Field(..., description="Type of data")
    total_weeks: int = Field(..., description="Total number of weeks available")
    mean: float = Field(..., description="Mean value")
    median: float = Field(..., description="Median value")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    latest_value: float = Field(..., description="Most recent value")
    latest_period: dict = Field(..., description="Most recent period information")
    trend: str = Field(..., description="Current trend: 'increasing' or 'decreasing'")

    class Config:
        schema_extra = {
            "example": {
                "commodity": "Cabbage",
                "data_type": "volume",
                "total_weeks": 24,
                "mean": 1250.5,
                "median": 1230.0,
                "std": 150.2,
                "min": 980.0,
                "max": 1520.0,
                "latest_value": 1300.0,
                "latest_period": {
                    "year": 2024,
                    "month": 12,
                    "week": 4,
                    "week_label": "Week 4 (Days 22-28)"
                },
                "trend": "increasing"
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    timestamp: str = Field(..., description="Current timestamp")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-12-23T10:30:00"
            }
        }


# ============================================================================
# Database Models (for reference/documentation)
# ============================================================================

class VolumeData(BaseModel):
    """Model representing weekly volume data in database"""
    id: Optional[int] = None
    year: int = Field(..., ge=2020, le=2100)
    month: int = Field(..., ge=1, le=12)
    week: int = Field(..., ge=1, le=5)
    commodity: str
    volume: float = Field(..., gt=0)
    encoded_by: str
    encoded_at: str

    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "month": 12,
                "week": 4,
                "commodity": "Cabbage",
                "volume": 1250.5,
                "encoded_by": "user123",
                "encoded_at": "2024-12-23T10:30:00"
            }
        }


class PriceData(BaseModel):
    """Model representing weekly price data in database"""
    id: Optional[int] = None
    year: int = Field(..., ge=2020, le=2100)
    month: int = Field(..., ge=1, le=12)
    week: int = Field(..., ge=1, le=5)
    commodity: str
    lowest_price: float = Field(..., gt=0)
    highest_price: float = Field(..., gt=0)
    average_price: float = Field(..., gt=0)
    encoded_by: str
    encoded_at: str

    @validator('highest_price')
    def validate_highest_price(cls, v, values):
        if 'lowest_price' in values and v < values['lowest_price']:
            raise ValueError("highest_price must be greater than or equal to lowest_price")
        return v

    @validator('average_price')
    def validate_average_price(cls, v, values):
        if 'lowest_price' in values and 'highest_price' in values:
            low = values['lowest_price']
            high = values['highest_price']
            if not (low <= v <= high):
                raise ValueError("average_price must be between lowest_price and highest_price")
        return v

    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "month": 12,
                "week": 4,
                "commodity": "Cabbage",
                "lowest_price": 25.0,
                "highest_price": 35.0,
                "average_price": 30.0,
                "encoded_by": "user123",
                "encoded_at": "2024-12-23T10:30:00"
            }
        }


# ============================================================================
# Error Response Models
# ============================================================================

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for client handling")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        schema_extra = {
            "example": {
                "detail": "Insufficient data for forecasting",
                "error_code": "INSUFFICIENT_DATA",
                "timestamp": "2024-12-23T10:30:00"
            }
        }