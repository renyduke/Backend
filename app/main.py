"""
main.py - Updated FastAPI Application with Pre-trained Model Support

CHANGES FROM ORIGINAL:
1. Added MODEL_CACHE and MODEL_METADATA global variables
2. Added startup_event() to load pre-trained models from train_external_data.py
3. Updated generate_forecast() to use pre-trained models when available
4. Added get_model_info() endpoint to check loaded models
5. Added fallback to database training if pre-trained model unavailable
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import shutil
import json
import asyncio
from supabase import create_client, Client

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .config import settings
from .models import (
    ForecastRequest, 
    ForecastResponse, 
    DashboardData, 
    CommoditiesResponse,
    HealthResponse
)
from .lstm_forecaster import LSTMForecaster, calculate_metrics
from .training_manager import TrainingManager
from .train_external_data import start_training

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

from fastapi.staticfiles import StaticFiles
# Create plots directory if not exists
os.makedirs(settings.PLOT_DIR, exist_ok=True)

# Mount plots directory
app.mount("/plots", StaticFiles(directory=settings.PLOT_DIR), name="plots")

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Create models directory
os.makedirs(settings.MODELS_DIR, exist_ok=True)

# Initialize Training Manager
training_manager = TrainingManager()

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
    """Scan for available models on startup but DO NOT load them into memory"""
    
    print("\n" + "="*70)
    print("SCANNING FOR MODELS (LAZY LOADING ENABLED)")
    print("="*70)
    
    if not os.path.exists(settings.MODELS_DIR):
        print(f"⚠ Model directory not found: {settings.MODELS_DIR}")
        return
    
    found_count = 0
    
    # Scan for pre-trained models
    for filename in os.listdir(settings.MODELS_DIR):
        if filename.endswith("_model.h5"):
            # Extract commodity and data_type from filename
                if not filename.startswith("global_"):
                    continue

                model_key = filename.replace("_model.h5", "")
                
                try:
                    # Load metadata ONLY
                    metadata_path = os.path.join(settings.MODELS_DIR, f"{model_key}_metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        MODEL_METADATA[model_key] = metadata
                        found_count += 1
                        
                        # Show simplified info
                        type_name = "Price" if "price" in model_key else "Volume"
                        print(f"  ✓ Found Unified Global {type_name} Model")
                    
                except Exception as e:
                    print(f"  ✗ Failed to read metadata for {model_key}: {e}")
    
    print(f"\n✅ System Ready with {found_count} Unified Global Models")
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


def fetch_all_supabase_data(table_name: str, select_cols: str = '*', eq_col: str = None, eq_val: str = None, order_cols: list = None):
    """Fetch all rows from a Supabase table using pagination to bypass max-rows limit."""
    # First get the exact total count
    count_query = supabase.table(table_name).select(select_cols, count='exact').limit(1)
    if eq_col and eq_val is not None:
        count_query = count_query.eq(eq_col, eq_val)
    count_response = count_query.execute()
    total_count = count_response.count
    print(f"[PAGINATION] {table_name}: total_count={total_count}")

    if not total_count:
        return []

    all_data = []
    page_size = 500  # Use smaller page size to avoid Supabase per-request cap
    offset = 0

    while offset < total_count:
        query = supabase.table(table_name).select(select_cols)
        if eq_col and eq_val is not None:
            query = query.eq(eq_col, eq_val)
        if order_cols:
            for col in order_cols:
                query = query.order(col)

        response = query.range(offset, offset + page_size - 1).execute()
        data = response.data
        print(f"[PAGINATION] {table_name}: offset={offset}, got={len(data)} rows")

        if not data:
            break

        all_data.extend(data)
        offset += len(data)

    print(f"[PAGINATION] {table_name}: total fetched={len(all_data)}")
    return all_data


def normalize_commodity_name(name: str) -> str:
    """Normalize commodity names to handle duplicates/typos (e.g., Cauli-flower -> Cauliflower)"""
    if not name:
        return name
    # Handle specific known duplicate "Cauli-flower"
    normalized = name.replace("Cauli-flower", "Cauliflower")
    return normalized


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
        order_columns = ['year', 'month', 'week']
        if data_type == 'volume':
            data = fetch_all_supabase_data('agri_volume', eq_col='commodity', eq_val=commodity, order_cols=order_columns)
            df = pd.DataFrame(data)
            if not df.empty:
                df['value'] = df['volume']
        else:  # price
            data = fetch_all_supabase_data('agri_price', eq_col='commodity', eq_val=commodity, order_cols=order_columns)
            df = pd.DataFrame(data)
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
# NEW: TRAINING ENDPOINTS
# ============================================================================

@app.post("/api/upload_dataset", tags=["Training"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset for training"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
            
        # Save file to root directory (replacing existing one or creating new)
        file_path = "vegetable_price_lstm_10000_structured.csv"  # Overwrite main file
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate CSV content
        try:
            df = pd.read_csv(file_path)
            
            # 1. Check required columns for the new weekly format
            required_columns = {'week', 'year', 'commodity', 'average_price'}
            
            missing = required_columns - set(df.columns)
            if missing:
                os.remove(file_path) # Delete invalid file
                raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}\nRequired: {', '.join(required_columns)}")
            
            # 2. Check row count
            if len(df) < 100:
                os.remove(file_path)
                raise HTTPException(status_code=400, detail=f"Dataset too small ({len(df)} rows). Need at least 100.")

            # 3. Check data types (basic check)
            if not pd.api.types.is_numeric_dtype(df['week']) or not pd.api.types.is_numeric_dtype(df['year']):
                os.remove(file_path)
                raise HTTPException(status_code=400, detail="'week' and 'year' must be numeric.")

            return {
                "message": "Dataset uploaded and validated successfully! ✅",
                "filename": file.filename,
                "rows": len(df),
                "commodities": df['commodity'].nunique(),
                "columns": list(df.columns)
            }
            
        except pd.errors.EmptyDataError:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")
        except Exception as validation_error:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Validation failed: {str(validation_error)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start_training", tags=["Training"])
async def trigger_training(background_tasks: BackgroundTasks):
    """Start the training process in background"""
    if training_manager.is_training:
        raise HTTPException(status_code=400, detail="Training is already in progress")
        
    try:
        # Start training in background thread via manager
        # We pass the default config, but you could accept config overrides here
        training_manager.start_training(start_training, None)
        return {"message": "Training started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket for streaming training logs"""
    await training_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages if any
            # We don't really expect messages from client, but we need to await something
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        training_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        training_manager.disconnect(websocket)

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


@app.get("/api/debug/models", tags=["Debug"])
async def debug_models():
    """Debug: show resolved paths and what model files exist"""
    import glob
    models_dir = settings.MODELS_DIR
    base_dir = settings.BASE_DIR
    cwd = os.getcwd()

    h5_files = glob.glob(os.path.join(models_dir, "*.h5"))
    pkl_files = glob.glob(os.path.join(models_dir, "*.pkl"))
    json_files = glob.glob(os.path.join(models_dir, "*.json"))

    return {
        "cwd": cwd,
        "base_dir": base_dir,
        "models_dir": models_dir,
        "models_dir_exists": os.path.exists(models_dir),
        "h5_files": [os.path.basename(f) for f in h5_files],
        "pkl_files": [os.path.basename(f) for f in pkl_files],
        "json_files": [os.path.basename(f) for f in json_files],
        "model_cache_keys": list(MODEL_CACHE.keys()),
        "model_metadata_keys": list(MODEL_METADATA.keys()),
    }



async def debug_counts():
    """Debug endpoint: check total row counts and year range in Supabase tables"""
    try:
        price_all = fetch_all_supabase_data('agri_price', select_cols='year')
        volume_all = fetch_all_supabase_data('agri_volume', select_cols='year')

        price_years = sorted(set(r['year'] for r in price_all)) if price_all else []
        volume_years = sorted(set(r['year'] for r in volume_all)) if volume_all else []

        return {
            "agri_price": {
                "total_fetched": len(price_all),
                "years": price_years,
            },
            "agri_volume": {
                "total_fetched": len(volume_all),
                "years": volume_years,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard", response_model=DashboardData, tags=["Dashboard"])
async def get_dashboard_data():
    """Get all data for dashboard"""
    try:
        order_columns = ['year', 'month', 'week']
        
        # Fetch volume data using pagination helper
        volume_data = fetch_all_supabase_data('agri_volume', order_cols=order_columns)
        
        # Fetch price data using pagination helper
        price_data = fetch_all_supabase_data('agri_price', order_cols=order_columns)
        
        # Format data with period information and normalize commodity names
        for item in volume_data:
            item['commodity'] = normalize_commodity_name(item['commodity'])
            item['period'] = create_period_key(item['year'], item['month'], item['week'])
            item['week_label'] = get_week_label(item['week'])
        
        for item in price_data:
            item['commodity'] = normalize_commodity_name(item['commodity'])
            item['period'] = create_period_key(item['year'], item['month'], item['week'])
            item['week_label'] = get_week_label(item['week'])
        
        # Get unique commodities (after normalization)
        commodities = sorted(list(set([item['commodity'] for item in volume_data + price_data])))
        
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
        # Fetch historical data from database for the requested commodity
        df = fetch_data_from_supabase(request.commodity, request.data_type)
        
        # Create model key for the unified global model
        model_key = f"global_{request.data_type}"
        local_model_key = f"local_{request.commodity}_{request.data_type}".lower().replace(" ", "_")
        
        # Check if pre-trained model exists (in cache or on disk)
        use_pretrained = False
        model_metadata = {}
        
        # 1. Check cache first
        if model_key in MODEL_CACHE:
            use_pretrained = True
            forecaster = MODEL_CACHE[model_key]
        # 2. Check disk if not in cache (Lazy Loading & Dynamic Loading)
        else:
            model_path_base = os.path.join(settings.MODELS_DIR, model_key)
            if os.path.exists(f"{model_path_base}_model.h5"):
                try:
                    print(f"?? Loading global model from disk dynamically: {model_key}")
                    forecaster = LSTMForecaster()
                    forecaster.load_model(model_path_base)
                    
                    # Cache it
                    MODEL_CACHE[model_key] = forecaster
                    
                    # Also load metadata if exists to keep it updated
                    metadata_path = f"{model_path_base}_metadata.json"
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            MODEL_METADATA[model_key] = json.load(f)
                            
                    use_pretrained = True
                except Exception as e:
                    print(f"? Failed to load global model {model_key}: {e}")
                    use_pretrained = False
        
        # Minimum weeks logic: Since a global model is trained, we only need `sequence_length` (4 weeks) as baseline input.
        # If no global model exists, we need 8 weeks to train a new local model on the fly.
        min_weeks = forecaster.sequence_length if use_pretrained else 8
        
        if df is None or len(df) < min_weeks:
            msg = (f"Insufficient historical data in Supabase database for '{request.commodity}'. "
                   f"The Global AI Model is loaded, but it requires at least the most recent {min_weeks} weeks of data "
                   f"as a starting point to generate future forecasts. Found: {len(df) if df is not None else 0}")
            raise HTTPException(status_code=400, detail=msg)
            
        # Prepare data
        values = df['value'].values
        
        if use_pretrained:
            print(f"?? Using pre-trained GLOBAL model for: {request.commodity} - {request.data_type}")
            model_metadata = MODEL_METADATA.get(model_key, {})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Global {request.data_type} model not found. Please upload a dataset and trigger training first."
            )
        
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
    
    # Refresh metadata dynamically by scanning the disk for newly trained models
    if os.path.exists(settings.MODELS_DIR):
        for filename in os.listdir(settings.MODELS_DIR):
            if filename.endswith("_model.h5") and filename.startswith("global_"):
                model_key = filename.replace("_model.h5", "")
                metadata_path = os.path.join(settings.MODELS_DIR, f"{model_key}_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            MODEL_METADATA[model_key] = json.load(f)
                    except Exception as e:
                        print(f"⚠ Failed to read dynamic metadata for {model_key}: {e}")
    
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
        volume_data = fetch_all_supabase_data('agri_volume', select_cols='commodity')
        price_data = fetch_all_supabase_data('agri_price', select_cols='commodity')
        
        commodities = set()
        for item in volume_data:
            commodities.add(normalize_commodity_name(item['commodity']))
        for item in price_data:
            commodities.add(normalize_commodity_name(item['commodity']))
        
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


# ============================================================================
# NEW: COMMODITY MANAGEMENT ENDPOINTS
# ============================================================================

@app.put("/api/commodities/{old_name}", tags=["Commodity Management"])
async def rename_commodity(old_name: str, payload: dict):
    """Rename a commodity across all data tables"""
    new_name = payload.get("new_name")
    if not new_name:
        raise HTTPException(status_code=400, detail="New name is required")
    
    try:
        # Update in agri_price table
        supabase.table('agri_price')\
            .update({'commodity': new_name})\
            .eq('commodity', old_name)\
            .execute()
            
        # Update in agri_volume table
        supabase.table('agri_volume')\
            .update({'commodity': new_name})\
            .eq('commodity', old_name)\
            .execute()
            
        return {"message": f"Successfully renamed '{old_name}' to '{new_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error renaming commodity: {str(e)}")


@app.delete("/api/commodities/{commodity_name}", tags=["Commodity Management"])
async def delete_commodity(commodity_name: str):
    """Delete all records associated with a commodity from all data tables"""
    try:
        # Delete from agri_price table
        supabase.table('agri_price')\
            .delete()\
            .eq('commodity', commodity_name)\
            .execute()
            
        # Delete from agri_volume table
        supabase.table('agri_volume')\
            .delete()\
            .eq('commodity', commodity_name)\
            .execute()
            
        return {"message": f"Successfully deleted all records for '{commodity_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting commodity: {str(e)}")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }