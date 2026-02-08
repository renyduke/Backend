"""
train_external_data.py

Script to train LSTM models using external datasets from Kaggle or other sources.
Run this script separately to pre-train models before starting your FastAPI server.

Usage:
    python train_external_data.py

Requirements:
    - Place your CSV file in the project root or update CSV_PATH
    - Ensure lstm_forecaster.py is in the same directory
    - Models will be saved to ./models/ directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
try:
    from .lstm_forecaster import LSTMForecaster, calculate_metrics
except ImportError:
    from lstm_forecaster import LSTMForecaster, calculate_metrics

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

CONFIG = {
    # Path to your downloaded CSV file
    'csv_path': 'vegetable_price_lstm_10000_structured.csv',  # CHANGE THIS to your file path
    
    # Column names in your CSV (adjust based on your dataset)
    'columns': {
        'date': 'date',           # Date column name
        'commodity': 'commodity',  # Commodity/Item column name
        'volume': 'volume',        # Volume/Quantity column (if available)
        'price': 'price_avg'       # Price column name
    },
    
    # Commodities to train (must match names in your CSV)
    'commodities': [
        'Cabbage',
        'Tomato',
        'Potato',
        'Carrots',
        'Green_Onion',
        'Lettuce',
        'Eggplant',
        'Cucumber',
        'Cauliflower',
        'Petchay',
        'Pepper',
        'Squash',
        'Sayote',
        'Broccoli',
        'Camote',
        'Ginger',
        'Beans',
        'Radish',
    ],
    
    # Training parameters
    'sequence_length': 4,  # Number of past weeks to use
    'epochs': 100,         # Training iterations
    'test_weeks': 8,       # Weeks to hold out for testing
    
    # Output directory
    'model_dir': 'models',
    'plot_dir': 'plots'
}


# ============================================================================
# STEP 1: Data Loading and Preparation
# ============================================================================

def load_csv_data(csv_path):
    """Load and display CSV structure"""
    print(f"\n{'='*70}")
    print("LOADING DATASET")
    print(f"{'='*70}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Please download a dataset and update CONFIG['csv_path']"
        )
    
    df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded: {csv_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nUnique commodities found:")
    commodity_col = CONFIG['columns']['commodity']
    if commodity_col in df.columns:
        commodities = df[commodity_col].unique()
        for i, comm in enumerate(sorted(commodities), 1):
            count = len(df[df[commodity_col] == comm])
            print(f"  {i}. {comm} ({count} records)")
    
    return df


def convert_to_weekly_data(df, commodity_name, value_type='volume'):
    """
    Convert dataset to weekly format matching your system structure
    
    Args:
        df: DataFrame with raw data
        commodity_name: Name of commodity to filter
        value_type: 'volume' or 'price'
    
    Returns:
        Dictionary with weekly data and metadata
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {commodity_name} - {value_type.upper()}")
    print(f"{'='*70}")
    
    # Get column names from config
    date_col = CONFIG['columns']['date']
    commodity_col = CONFIG['columns']['commodity']
    
    if value_type == 'volume':
        value_col = CONFIG['columns']['volume']
    else:
        value_col = CONFIG['columns']['price']
    
    # Filter for specific commodity
    df_filtered = df[
        df[commodity_col].str.contains(commodity_name, case=False, na=False)
    ].copy()
    
    if len(df_filtered) == 0:
        raise ValueError(f"No data found for commodity: {commodity_name}")
    
    print(f"  Found {len(df_filtered)} records")
    
    # Convert to datetime
    df_filtered['Date'] = pd.to_datetime(df_filtered[date_col], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['Date'])
    
    # Extract time components
    df_filtered['Year'] = df_filtered['Date'].dt.year
    df_filtered['Month'] = df_filtered['Date'].dt.month
    df_filtered['Day'] = df_filtered['Date'].dt.day
    
    # Calculate week of month (1-5)
    df_filtered['Week'] = ((df_filtered['Day'] - 1) // 7) + 1
    df_filtered['Week'] = df_filtered['Week'].clip(upper=5)
    
    # Aggregate to weekly data
    aggregation = 'mean' if value_type == 'price' else 'sum'
    
    weekly_data = df_filtered.groupby(['Year', 'Month', 'Week']).agg({
        value_col: aggregation
    }).reset_index()
    
    # Sort chronologically
    weekly_data = weekly_data.sort_values(['Year', 'Month', 'Week'])
    weekly_data = weekly_data.reset_index(drop=True)
    
    # Create period labels
    weekly_data['period'] = (
        weekly_data['Year'].astype(str) + '-' + 
        weekly_data['Month'].astype(str).str.zfill(2) + '-' + 
        weekly_data['Week'].astype(str).str.zfill(2)
    )
    
    values = weekly_data[value_col].values
    
    print(f"  Converted to {len(weekly_data)} weekly data points")
    print(f"  Date range: {weekly_data.iloc[0]['period']} to {weekly_data.iloc[-1]['period']}")
    print(f"  Value range: {values.min():.2f} to {values.max():.2f}")
    print(f"  Mean value: {values.mean():.2f}")
    print(f"  Std deviation: {values.std():.2f}")
    
    return {
        'values': values,
        'weekly_data': weekly_data,
        'metadata': {
            'commodity': commodity_name,
            'data_type': value_type,
            'total_weeks': len(weekly_data),
            'date_range': {
                'start': weekly_data.iloc[0]['period'],
                'end': weekly_data.iloc[-1]['period']
            },
            'statistics': {
                'min': float(values.min()),
                'max': float(values.max()),
                'mean': float(values.mean()),
                'std': float(values.std())
            }
        }
    }


# ============================================================================
# STEP 2: Model Training
# ============================================================================

def train_lstm_model(data_dict, commodity_name, data_type):
    """
    Train LSTM model with prepared data
    
    Args:
        data_dict: Dictionary from convert_to_weekly_data()
        commodity_name: Name of commodity
        data_type: 'volume' or 'price'
    
    Returns:
        Dictionary with trained model and results
    """
    values = data_dict['values']
    
    # Check data sufficiency
    min_required = CONFIG['sequence_length'] + CONFIG['test_weeks'] + 2
    if len(values) < min_required:
        raise ValueError(
            f"Insufficient data. Need at least {min_required} weeks, "
            f"but only have {len(values)} weeks."
        )
    
    # Split data
    test_weeks = CONFIG['test_weeks']
    train_data = values[:-test_weeks]
    test_data = values[-test_weeks:]
    
    print(f"\nDataset Split:")
    print(f"  Training: {len(train_data)} weeks")
    print(f"  Testing: {len(test_data)} weeks")
    
    # Initialize forecaster
    forecaster = LSTMForecaster(sequence_length=CONFIG['sequence_length'])
    
    print(f"\nTraining LSTM Model...")
    print(f"  Sequence length: {CONFIG['sequence_length']} weeks")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Starting training...")
    
    # Train model
    history = forecaster.train(
        train_data, 
        epochs=CONFIG['epochs'], 
        verbose=1
    )
    
    # Generate predictions
    print(f"\nGenerating predictions...")
    predictions = forecaster.forecast(train_data, weeks_ahead=test_weeks)
    
    # Calculate metrics
    metrics = calculate_metrics(test_data, predictions)
    
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"  Mean Absolute Error (MAE):     {metrics['mae']:>10.2f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:>10.2f}")
    print(f"  Mean Absolute % Error (MAPE):   {metrics['mape']:>10.2f}%")
    print(f"  Accuracy:                       {metrics['accuracy']:>10.2f}%")
    print(f"{'='*70}")
    
    return {
        'forecaster': forecaster,
        'metrics': metrics,
        'predictions': predictions,
        'test_data': test_data,
        'train_data': train_data,
        'history': history
    }


# ============================================================================
# STEP 3: Visualization
# ============================================================================

def create_visualizations(results, data_dict, commodity_name, data_type):
    """Create and save training visualizations"""
    
    # Create plots directory
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)
    
    train_data = results['train_data']
    test_data = results['test_data']
    predictions = results['predictions']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full forecast
    train_weeks = range(len(train_data))
    test_weeks = range(len(train_data), len(train_data) + len(test_data))
    
    ax1.plot(train_weeks, train_data, 'b-', label='Training Data', linewidth=2)
    ax1.plot(test_weeks, test_data, 'g-', label='Actual (Test)', 
             linewidth=2, marker='o', markersize=6)
    ax1.plot(test_weeks, predictions, 'r--', label='Predictions', 
             linewidth=2, marker='s', markersize=6)
    
    ax1.set_title(f'{commodity_name} - {data_type.upper()} Forecast', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Week Number', fontsize=11)
    ax1.set_ylabel(f'{data_type.capitalize()}', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss
    if 'history' in results and hasattr(results['history'], 'history'):
        loss = results['history'].history['loss']
        epochs = range(1, len(loss) + 1)
        
        ax2.plot(epochs, loss, 'b-', linewidth=2)
        ax2.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss (MSE)', fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{CONFIG['plot_dir']}/{commodity_name.lower()}_{data_type}_training.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {filename}")
    
    plt.close()


# ============================================================================
# STEP 4: Save Models and Metadata
# ============================================================================

def save_model_and_metadata(results, data_dict, commodity_name, data_type):
    """Save trained model, scaler, and metadata"""
    
    # Create models directory
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    # Base path for saving
    model_path = f"{CONFIG['model_dir']}/{commodity_name.lower()}_{data_type}"
    
    # Save LSTM model and scaler
    forecaster = results['forecaster']
    forecaster.save_model(model_path)
    
    # Save comprehensive metadata
    metadata = {
        'commodity': commodity_name,
        'data_type': data_type,
        'training_date': datetime.now().isoformat(),
        'model_config': {
            'sequence_length': CONFIG['sequence_length'],
            'epochs': CONFIG['epochs'],
            'architecture': 'LSTM_Weekly'
        },
        'data_info': data_dict['metadata'],
        'performance_metrics': results['metrics'],
        'training_info': {
            'train_weeks': len(results['train_data']),
            'test_weeks': len(results['test_data'])
        }
    }
    
    metadata_path = f"{model_path}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Model saved: {model_path}_model.h5")
    print(f"✓ Scaler saved: {model_path}_scaler.pkl")
    print(f"✓ Metadata saved: {metadata_path}")
    
    return model_path


# ============================================================================
# STEP 5: Batch Training for Multiple Commodities
# ============================================================================

def batch_train_all_commodities(df):
    """Train models for all configured commodities"""
    
    print(f"\n{'='*70}")
    print("BATCH TRAINING - ALL COMMODITIES")
    print(f"{'='*70}")
    print(f"Commodities to train: {len(CONFIG['commodities'])}")
    print(f"Data types: volume, price")
    print(f"Total models to train: {len(CONFIG['commodities']) * 2}")
    
    results_summary = []
    
    for i, commodity in enumerate(CONFIG['commodities'], 1):
        print(f"\n\n{'#'*70}")
        print(f"COMMODITY {i}/{len(CONFIG['commodities'])}: {commodity}")
        print(f"{'#'*70}")
        
        # Train for both volume and price
        for data_type in [ 'price','volume']:
            try:
                # Prepare data
                data_dict = convert_to_weekly_data(df, commodity, data_type)
                
                # Train model
                train_results = train_lstm_model(data_dict, commodity, data_type)
                
                # Create visualizations
                create_visualizations(train_results, data_dict, commodity, data_type)
                
                # Save model
                model_path = save_model_and_metadata(
                    train_results, data_dict, commodity, data_type
                )
                
                # Record success
                results_summary.append({
                    'commodity': commodity,
                    'data_type': data_type,
                    'status': 'SUCCESS',
                    'accuracy': train_results['metrics']['accuracy'],
                    'mae': train_results['metrics']['mae'],
                    'weeks_trained': len(data_dict['values'])
                })
                
                print(f"\n✅ SUCCESS: {commodity} - {data_type}")
                
            except Exception as e:
                print(f"\n❌ FAILED: {commodity} - {data_type}")
                print(f"   Error: {str(e)}")
                
                results_summary.append({
                    'commodity': commodity,
                    'data_type': data_type,
                    'status': 'FAILED',
                    'error': str(e)
                })
    
    return results_summary


def print_training_summary(results_summary):
    """Print final training summary"""
    
    print(f"\n\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in results_summary if r['status'] == 'SUCCESS']
    failed = [r for r in results_summary if r['status'] == 'FAILED']
    
    print(f"\nTotal models trained: {len(results_summary)}")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n{'='*70}")
        print("SUCCESSFUL MODELS")
        print(f"{'='*70}")
        print(f"{'Commodity':<15} {'Type':<8} {'Accuracy':<12} {'MAE':<10} {'Weeks'}")
        print("-" * 70)
        for r in successful:
            print(f"{r['commodity']:<15} {r['data_type']:<8} "
                  f"{r['accuracy']:>6.2f}%     {r['mae']:>8.2f}  {r['weeks_trained']:>5}")
    
    if failed:
        print(f"\n{'='*70}")
        print("FAILED MODELS")
        print(f"{'='*70}")
        for r in failed:
            print(f"❌ {r['commodity']} - {r['data_type']}")
            print(f"   Error: {r['error']}")
    
    # Save summary to JSON
    summary_path = f"{CONFIG['model_dir']}/training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': CONFIG,
            'results': results_summary
        }, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_path}")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Global logger function
def log(message, callback=None):
    if callback:
        callback(message)
    else:
        print(message)

def start_training(config=None, log_callback=None):
    """
    Start the training process with optional config override and log callback.
    config: dict (optional) - Override checking CONFIG
    log_callback: function(str) - Function to receive log messages
    """
    global CONFIG
    
    # Use provided config or default
    active_config = config if config else CONFIG

    # Update global config if needed (for other functions using it)
    if config:
        CONFIG.update(config)

    log("\n", log_callback)
    log("="*70, log_callback)
    log(" "*15 + "LSTM MODEL TRAINING FROM EXTERNAL DATA", log_callback)
    log("="*70, log_callback)
    
    try:
        log(f"Starting training with config: {active_config.get('csv_path', 'default')}", log_callback)
        
        # Step 1: Load dataset
        # We need to modify helper functions to use the logger, 
        # but for now we'll just redirect their stdout if possible or accept they print to console
        # A better approach is to pass the logger down, but that requires rewriting all functions.
        # For this implementation, we will assume helper functions print to console, 
        # and we will log major steps here.
        
        # Reloading data with new path if specified
        csv_path = active_config['csv_path']
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"CSV file not found: {csv_path}")

        log(f"Loading dataset from {csv_path}...", log_callback)
        df = pd.read_csv(csv_path)
        log(f"Dataset loaded: {len(df)} rows", log_callback)
        
        # Step 2: Batch train all commodities
        # We'll reimplement the loop here to log progress
        log("\n" + "="*70, log_callback)
        log("BATCH TRAINING - ALL COMMODITIES", log_callback)
        log("="*70, log_callback)
        
        commodities = active_config['commodities']
        results_summary = []
        
        total_steps = len(commodities) * 2
        current_step = 0
        
        for i, commodity in enumerate(commodities, 1):
            log(f"\nProcessing Commodity {i}/{len(commodities)}: {commodity}", log_callback)
            
            for data_type in ['price', 'volume']:
                current_step += 1
                try:
                    log(f"  Training {commodity} - {data_type} ({current_step}/{total_steps})...", log_callback)
                    
                    # Prepare data
                    data_dict = convert_to_weekly_data(df, commodity, data_type)
                    
                    # Train model
                    train_results = train_lstm_model(data_dict, commodity, data_type)
                    
                    # Create visualizations
                    create_visualizations(train_results, data_dict, commodity, data_type)
                    
                    # Save model
                    save_model_and_metadata(train_results, data_dict, commodity, data_type)
                    
                    # Record success
                    results_summary.append({
                        'commodity': commodity,
                        'data_type': data_type,
                        'status': 'SUCCESS',
                        'accuracy': train_results['metrics']['accuracy'],
                        'mae': train_results['metrics']['mae'],
                        'weeks_trained': len(data_dict['values'])
                    })
                    
                    log(f"  ✅ SUCCESS: {commodity} - {data_type} (Acc: {train_results['metrics']['accuracy']:.2f}%)", log_callback)
                    
                except Exception as e:
                    log(f"  ❌ FAILED: {commodity} - {data_type}", log_callback)
                    log(f"     Error: {str(e)}", log_callback)
                    
                    results_summary.append({
                        'commodity': commodity,
                        'data_type': data_type,
                        'status': 'FAILED',
                        'error': str(e)
                    })
        
        # Step 3: Summary
        log("\n" + "="*70, log_callback)
        log("TRAINING SUMMARY", log_callback)
        log("="*70, log_callback)
        
        successful = [r for r in results_summary if r['status'] == 'SUCCESS']
        failed = [r for r in results_summary if r['status'] == 'FAILED']
        
        log(f"Total trained: {len(results_summary)}", log_callback)
        log(f"Successful: {len(successful)}", log_callback)
        log(f"Failed: {len(failed)}", log_callback)
        
        # Save summary to JSON
        summary_path = f"{active_config['model_dir']}/training_summary.json"
        
        # Create output dir if it doesn't exist
        os.makedirs(active_config['model_dir'], exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': active_config,
                'results': results_summary
            }, f, indent=2)
        
        log(f"\nSummary saved to {summary_path}", log_callback)
        log("Training process completed successfully.", log_callback)
        
        return results_summary
        
    except Exception as e:
        log(f"\n❌ FATAL ERROR: {str(e)}", log_callback)
        import traceback
        log(traceback.format_exc(), log_callback)
        raise e

def main():
    start_training()

if __name__ == "__main__":
    main()