"""
main.py - Updated FastAPI Application with Pre-trained Model Support

CHANGES FROM ORIGINAL:
1. Added MODEL_CACHE and MODEL_METADATA global variables
2. Added startup_event() to load pre-trained models from train_external_data.py
3. Updated generate_forecast() to use pre-trained models when available
4. Added get_model_info() endpoint to check loaded models
5. Added fallback to database training if pre-trained model unavailable
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import os
import json
from supabase import create_client, Client

from .config import settings
from .models import (
    ForecastRequest, 
    ForecastResponse, 
    DashboardData, 
    CommoditiesResponse,
    HealthResponse
)
from .lstm_forecaster import LSTMForecaster, calculate_metrics

# Validate settings
settings.validate()

# Initialize FastAPI
app = FastAPI(
    title="AgriData Forecasting API",
    description="LSTM-based forecasting for agricultural data (Weekly)",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Create models directory
os.makedirs(settings.MODELS_DIR, exist_ok=True)

# ============================================================================
# NEW: GLOBAL MODEL CACHE FOR PRE-TRAINED MODELS
# ============================================================================

MODEL_CACHE = {}  # Store loaded pre-trained models
MODEL_METADATA = {}  # Store model metadata (accuracy, training date, etc.)


# ============================================================================
# NEW: STARTUP EVENT - Load Pre-trained Models
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load pre-trained models from train_external_data.py on startup"""
    
    print("\n" + "="*70)
    print("LOADING PRE-TRAINED MODELS")
    print("="*70)
    
    if not os.path.exists(settings.MODELS_DIR):
        print(f"⚠ Model directory not found: {settings.MODELS_DIR}")
        print("  Models will be trained on-demand from database")
        print("  Run train_external_data.py to create pre-trained models")
        return
    
    loaded_count = 0
    
    # Scan for pre-trained models
    for filename in os.listdir(settings.MODELS_DIR):
        if filename.endswith("_model.h5"):
            # Extract commodity and data_type from filename
            # Format: commodity_datatype_model.h5
            base_name = filename.replace("_model.h5", "")
            parts = base_name.rsplit("_", 1)
            
            if len(parts) == 2:
                commodity, data_type = parts
                model_key = f"{commodity}_{data_type}"
                
                try:
                    # Load model
                    model_path = os.path.join(settings.MODELS_DIR, base_name)
                    forecaster = LSTMForecaster()
                    forecaster.load_model(model_path)
                    
                    # Load metadata if available
                    metadata_path = f"{model_path}_metadata.json"
                    metadata = None
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        MODEL_METADATA[model_key] = metadata
                    
                    MODEL_CACHE[model_key] = forecaster
                    loaded_count += 1
                    
                    # Show model info
                    acc = metadata.get('performance_metrics', {}).get('accuracy', 'N/A')
                    print(f"  ✓ {commodity.capitalize():<12} {data_type:<8} (Accuracy: {acc if isinstance(acc, str) else f'{acc:.1f}%'})")
                    
                except Exception as e:
                    print(f"  ✗ Failed to load {model_key}: {e}")
    
    if loaded_count > 0:
        print(f"\n✅ Loaded {loaded_count} pre-trained models")
    else:
        print("\n⚠ No pre-trained models found")
        print("  Models will be trained on-demand from database")
    
    print("="*70 + "\n")


# ============================================================================
# EXISTING HELPER FUNCTIONS (unchanged)
# ============================================================================

def get_week_label(week: int) -> str:
    """Convert week number to readable label"""
    labels = {
        1: "Week 1 (Days 1-7)",
        2: "Week 2 (Days 8-14)",
        3: "Week 3 (Days 15-21)",
        4: "Week 4 (Days 22-28)",
        5: "Week 5 (Days 29-31)"
    }
    return labels.get(week, f"Week {week}")


def increment_week(year: int, month: int, week: int) -> tuple:
    """Increment week and handle month/year transitions"""
    week += 1
    if week > 5:
        week = 1
        month += 1
        if month > 12:
            month = 1
            year += 1
    return year, month, week


def create_period_key(year: int, month: int, week: int) -> str:
    """Create a sortable period key for ordering"""
    return f"{year:04d}-{month:02d}-{week:01d}"


def fetch_data_from_supabase(commodity: str, data_type: str):
    """Fetch historical weekly data from Supabase"""
    try:
        if data_type == 'volume':
            response = supabase.table('agri_volume')\
                .select('*')\
                .eq('commodity', commodity)\
                .order('year')\
                .order('month')\
                .order('week')\
                .execute()
            df = pd.DataFrame(response.data)
            if not df.empty:
                df['value'] = df['volume']
        else:  # price
            response = supabase.table('agri_price')\
                .select('*')\
                .eq('commodity', commodity)\
                .order('year')\
                .order('month')\
                .order('week')\
                .execute()
            df = pd.DataFrame(response.data)
            if not df.empty:
                df['value'] = df['average_price']
        
        if df.empty:
            return None
        
        # Create period key for proper sorting
        df['period'] = df.apply(lambda row: create_period_key(row['year'], row['month'], row['week']), axis=1)
        df = df.sort_values('period')
        
        # Add week label for display
        df['week_label'] = df['week'].apply(get_week_label)
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# ============================================================================
# EXISTING ENDPOINTS (unchanged)
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "AgriData Forecasting API",
        "version": "2.0.0",
        "data_structure": "weekly",
        "docs": "/docs"
    }


@app.get("/api/dashboard", response_model=DashboardData, tags=["Dashboard"])
async def get_dashboard_data():
    """Get all data for dashboard"""
    try:
        # Fetch volume data
        volume_response = supabase.table('agri_volume')\
            .select('*')\
            .order('year')\
            .order('month')\
            .order('week')\
            .execute()
        volume_data = volume_response.data
        
        # Fetch price data
        price_response = supabase.table('agri_price')\
            .select('*')\
            .order('year')\
            .order('month')\
            .order('week')\
            .execute()
        price_data = price_response.data
        
        # Get unique commodities
        commodities = list(set([item['commodity'] for item in volume_data + price_data]))
        commodities.sort()
        
        # Format data with period information
        for item in volume_data:
            item['period'] = create_period_key(item['year'], item['month'], item['week'])
            item['week_label'] = get_week_label(item['week'])
        
        for item in price_data:
            item['period'] = create_period_key(item['year'], item['month'], item['week'])
            item['week_label'] = get_week_label(item['week'])
        
        # Get period range
        all_periods = [item['period'] for item in volume_data + price_data]
        period_range = {
            "start": min(all_periods) if all_periods else None,
            "end": max(all_periods) if all_periods else None
        }
        
        return {
            "volume_data": volume_data,
            "price_data": price_data,
            "commodities": commodities,
            "period_range": period_range
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UPDATED: FORECAST ENDPOINT - Now uses pre-trained models when available
# ============================================================================

@app.post("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def generate_forecast(request: ForecastRequest):
    """Generate forecast using LSTM for future weeks"""
    try:
        # Create model key
        model_key = f"{request.commodity.lower()}_{request.data_type}"
        
        # Check if pre-trained model exists
        use_pretrained = model_key in MODEL_CACHE
        
        if use_pretrained:
            print(f"📦 Using pre-trained model: {request.commodity} - {request.data_type}")
            forecaster = MODEL_CACHE[model_key]
            model_metadata = MODEL_METADATA.get(model_key, {})
        else:
            print(f"🔨 Training new model from database: {request.commodity} - {request.data_type}")
        
        # Fetch historical data from database (needed for both pre-trained and new models)
        df = fetch_data_from_supabase(request.commodity, request.data_type)
        
        min_weeks = 8
        if df is None or len(df) < min_weeks:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data for forecasting. Need at least {min_weeks} weeks. Found: {len(df) if df is not None else 0}"
            )
        
        # Prepare data
        values = df['value'].values
        
        # If no pre-trained model, train from database
        if not use_pretrained:
            forecaster = LSTMForecaster(sequence_length=4)
            forecaster.train(values, epochs=50, verbose=0)
            
            # Optionally save this model for future use
            model_path = f"{settings.MODELS_DIR}/{request.commodity.lower()}_{request.data_type}"
            forecaster.save_model(model_path)
            print(f"  ✓ Model saved: {model_path}")
        
        # Generate forecasts
        forecast_values = forecaster.forecast(values, request.weeks_ahead)
        
        # Get last period info
        last_year = df.iloc[-1]['year']
        last_month = df.iloc[-1]['month']
        last_week = df.iloc[-1]['week']
        
        # Generate future periods
        forecast_periods = []
        current_year, current_month, current_week = last_year, last_month, last_week
        
        for _ in range(request.weeks_ahead):
            current_year, current_month, current_week = increment_week(
                current_year, current_month, current_week
            )
            forecast_periods.append({
                'year': current_year,
                'month': current_month,
                'week': current_week,
                'week_label': get_week_label(current_week),
                'period': create_period_key(current_year, current_month, current_week)
            })
        
        # Calculate metrics
        if use_pretrained and model_key in MODEL_METADATA:
            # Use pre-computed metrics from training
            metrics = MODEL_METADATA[model_key].get('performance_metrics', {})
        else:
            # Calculate metrics on last 4 weeks if enough data
            if len(values) >= 12:
                test_values = values[-4:]
                forecaster_test = LSTMForecaster(sequence_length=4)
                forecaster_test.train(values[:-4], epochs=50, verbose=0)
                test_predictions = forecaster_test.forecast(values[:-4], 4)
                metrics = calculate_metrics(test_values, test_predictions)
            else:
                metrics = {"note": "Insufficient data for validation metrics"}
        
        # Format historical data
        historical_data = [
            {
                "year": int(row['year']),
                "month": int(row['month']),
                "week": int(row['week']),
                "week_label": row['week_label'],
                "period": row['period'],
                "value": float(row['value'])
            }
            for _, row in df.iterrows()
        ]
        
        # Format forecast data
        forecast_data = [
            {
                "year": period['year'],
                "month": period['month'],
                "week": period['week'],
                "week_label": period['week_label'],
                "period": period['period'],
                "value": float(value)
            }
            for period, value in zip(forecast_periods, forecast_values)
        ]
        
        return {
            "commodity": request.commodity,
            "data_type": request.data_type,
            "historical_data": historical_data,
            "forecast_data": forecast_data,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


# ============================================================================
# NEW: MODEL INFO ENDPOINT - Check which models are loaded
# ============================================================================

@app.get("/api/models", tags=["Models"])
async def get_model_info():
    """Get information about loaded pre-trained models"""
    models_info = []
    
    for model_key, metadata in MODEL_METADATA.items():
        commodity, data_type = model_key.rsplit('_', 1)
        
        info = {
            "commodity": commodity.capitalize(),
            "data_type": data_type,
            "model_key": model_key,
            "is_loaded": model_key in MODEL_CACHE,
            "training_date": metadata.get('training_date', 'Unknown'),
            "performance": metadata.get('performance_metrics', {}),
            "data_info": metadata.get('data_info', {})
        }
        models_info.append(info)
    
    return {
        "total_models": len(MODEL_CACHE),
        "models": models_info,
        "note": "Run train_external_data.py to create more pre-trained models"
    }


# ============================================================================
# EXISTING ENDPOINTS (unchanged)
# ============================================================================

@app.get("/api/commodities", response_model=CommoditiesResponse, tags=["Data"])
async def get_commodities():
    """Get list of available commodities"""
    try:
        volume_response = supabase.table('agri_volume').select('commodity').execute()
        price_response = supabase.table('agri_price').select('commodity').execute()
        
        commodities = set()
        for item in volume_response.data:
            commodities.add(item['commodity'])
        for item in price_response.data:
            commodities.add(item['commodity'])
        
        return {"commodities": sorted(list(commodities))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics/{commodity}/{data_type}", tags=["Data"])
async def get_statistics(commodity: str, data_type: str):
    """Get statistical summary for a commodity"""
    try:
        df = fetch_data_from_supabase(commodity, data_type)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {commodity}")
        
        values = df['value'].values
        
        stats = {
            "commodity": commodity,
            "data_type": data_type,
            "total_weeks": len(values),
            "mean": float(values.mean()),
            "median": float(pd.Series(values).median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "latest_value": float(values[-1]),
            "latest_period": {
                "year": int(df.iloc[-1]['year']),
                "month": int(df.iloc[-1]['month']),
                "week": int(df.iloc[-1]['week']),
                "week_label": df.iloc[-1]['week_label']
            },
            "trend": "increasing" if len(values) > 1 and values[-1] > values[-2] else "decreasing"
        }
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }