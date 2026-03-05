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
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from datetime import datetime
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    from .lstm_forecaster import LSTMForecaster, calculate_metrics
except ImportError:
    from lstm_forecaster import LSTMForecaster, calculate_metrics

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

CONFIG = {
    # Per-data-type settings
    'price': {
        'csv_path': 'data_testing_with_month_int.csv',
        'columns': {
            'commodity': 'commodity',
            'value': 'average_price',
            'year': 'year',
            'month': 'month',
            'week': 'week'
        }
    },
    'volume': {
        'csv_path': 'app/agricultural_data.csv',
        'columns': {
            'commodity': 'Commodity',
            'value': 'Volume',
            'date': 'Date'  # Use date string instead of Y/M/W columns
        }
    },
    
    # Global commodities (can be overridden per-run)
    'commodities': [
        'Cabbage', 'Tomato', 'Potato', 'Carrots', 'Green_Onion',
        'Lettuce', 'Eggplant', 'Cucumber', 'Cauliflower', 'Petchay',
        'Pepper', 'Squash', 'Sayote', 'Broccoli', 'Camote', 'Ginger',
        'Beans', 'Radish'
    ],
    
    # Training parameters
    'sequence_length': 4,
    'epochs': 100,
    'test_weeks': 8,
    
    # Output directory
    'model_dir': 'models',
    'plot_dir': 'plots'
}


# ============================================================================
# STEP 1: Data Loading and Preparation
# ============================================================================

def load_csv_data(csv_path, data_type='price'):
    """Load and display CSV structure"""
    print(f"\n{'='*70}")
    print(f"LOADING {data_type.upper()} DATASET")
    print(f"{'='*70}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded: {csv_path} ({len(df):,} rows)")
    
    return df


def convert_to_weekly_data(df, commodity_name, value_type='volume'):
    """
    Convert dataset to weekly format matching your system structure.
    Now supports both explicit Y/M/W columns and Date string columns.
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {commodity_name} - {value_type.upper()}")
    print(f"{'='*70}")
    
    # Get configuration for specific data type
    type_config = CONFIG.get(value_type, {})
    col_map = type_config.get('columns', {})
    
    commodity_col = col_map.get('commodity', 'commodity')
    value_col = col_map.get('value', 'value')
    
    # Filter for specific commodity
    # Standardize commodity names for search (handle underscores/spaces)
    search_name = commodity_name.replace('_', ' ')
    df_filtered = df[
        df[commodity_col].astype(str).str.contains(search_name, case=False, na=False) |
        df[commodity_col].astype(str).str.contains(commodity_name, case=False, na=False)
    ].copy()
    
    if len(df_filtered) == 0:
        raise ValueError(f"No data found for commodity: {search_name}")
    
    print(f"  Found {len(df_filtered)} records")
    
    # Handle Date Parsing if Y/M/W columns are missing
    if 'date' in col_map and col_map['date'] in df_filtered.columns:
        date_col = col_map['date']
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        df_filtered['Year'] = df_filtered[date_col].dt.year
        df_filtered['Month'] = df_filtered[date_col].dt.month
        # Logic for Week of Month (1-5)
        df_filtered['Week'] = df_filtered[date_col].dt.day.apply(lambda d: min(5, (d - 1) // 7 + 1))
    else:
        # Use explicit mapping
        df_filtered['Year'] = df_filtered[col_map.get('year', 'year')]
        df_filtered['Month'] = df_filtered[col_map.get('month', 'month')]
        df_filtered['Week'] = df_filtered[col_map.get('week', 'week')]
    
    # Drop rows with missing essential data
    df_filtered = df_filtered.dropna(subset=['Year', 'Month', 'Week', value_col])
    
    # Aggregate to weekly data
    aggregation = 'mean' if value_type == 'price' else 'sum'
    
    weekly_data = df_filtered.groupby(['Year', 'Month', 'Week']).agg({
        value_col: aggregation
    }).reset_index()
    
    # Sort chronologically
    weekly_data = weekly_data.sort_values(['Year', 'Month', 'Week'])
    weekly_data = weekly_data.reset_index(drop=True)
    
    # Create period labels (YYYY-MM-WW)
    weekly_data['period'] = (
        weekly_data['Year'].astype(str) + '-' + 
        weekly_data['Month'].astype(str).str.zfill(2) + '-' + 
        weekly_data['Week'].astype(str).str.zfill(2)
    )
    
    values = weekly_data[value_col].values
    
    print(f"  Converted to {len(weekly_data)} weekly data points")
    print(f"  Date range: {weekly_data.iloc[0]['period']} to {weekly_data.iloc[-1]['period']}")
    print(f"  Value range: {values.min():.2f} to {values.max():.2f}")
    
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

def batch_train_all_commodities():
    """Train models for all configured commodities across all data types"""
    
    print(f"\n{'='*70}")
    print("BATCH TRAINING - ALL COMMODITIES")
    print(f"{'='*70}")
    
    results_summary = []
    
    for data_type in ['price', 'volume']:
        type_config = CONFIG.get(data_type)
        if not type_config:
            continue
            
        csv_path = type_config['csv_path']
        if not os.path.exists(csv_path):
            print(f"Skipping {data_type}: CSV not found at {csv_path}")
            continue
            
        print(f"\nLoading {data_type.upper()} dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        
        for i, commodity in enumerate(CONFIG['commodities'], 1):
            print(f"\n--- {data_type.upper()} | {commodity} ({i}/{len(CONFIG['commodities'])}) ---")
            
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
                print(f"✅ SUCCESS")
                
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
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

def archive_old_models(model_dir):
    """Move non-global models to an archive folder"""
    archive_dir = os.path.join(model_dir, 'archive')
    os.makedirs(archive_dir, exist_ok=True)
    
    for filename in os.listdir(model_dir):
        if filename == 'archive' or filename == 'training_summary.json':
            continue
            
        if not filename.startswith('global_'):
            src = os.path.join(model_dir, filename)
            dst = os.path.join(archive_dir, filename)
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            except Exception as e:
                print(f"Warning: Could not archive {filename}: {e}")

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
        log(f"Starting unified training flow...", log_callback)
        
        results_summary = []
        global_commodities = active_config.get('commodities', [])

        for data_type in ['price', 'volume']:
            log(f"\nProcessing {data_type.upper()}...", log_callback)
            
            # Load dataset for this specific data type
            type_config = active_config.get(data_type, {})
            csv_path = type_config.get('csv_path')
            col_map = type_config.get('columns', {})
            
            if not csv_path or not os.path.exists(csv_path):
                 log(f"  ✗ CSV for {data_type} not found: {csv_path}", log_callback)
                 continue

            log(f"  Loading dataset: {csv_path}...", log_callback)
            df = pd.read_csv(csv_path)
            
            commodity_col = col_map.get('commodity', 'commodity')
            
            all_train_data = []
            test_data_dict = {}
            data_dicts = {}
            
            # Filter global commodities to only those present in THIS dataset
            available_in_csv = df[commodity_col].unique() if commodity_col in df.columns else []
            
            for i, commodity in enumerate(global_commodities, 1):
                try:
                    # Robust commodity search (handle underscores vs spaces)
                    search_name = commodity.replace('_', ' ')
                    
                    found_match = any(c.lower() == search_name.lower() or c.lower() == commodity.lower() 
                                    for c in available_in_csv)
                                    
                    if not found_match:
                        continue

                    # Prepare data
                    data_dict = convert_to_weekly_data(df, commodity, data_type)
                    values = data_dict['values']
                    
                    # More adaptive data requirements for smaller datasets
                    test_weeks = active_config['test_weeks']
                    seq_len = active_config['sequence_length']
                    
                    # If dataset is too small, reduce requirements
                    if len(values) < (seq_len + test_weeks + 2):
                        test_weeks = max(2, len(values) // 4)
                        seq_len = min(seq_len, len(values) - test_weeks - 2)
                        
                    if len(values) >= (seq_len + test_weeks + 2):
                        train_data = values[:-test_weeks]
                        test_data = values[-test_weeks:]
                        
                        all_train_data.append(train_data)
                        test_data_dict[commodity] = {
                            'test_data': test_data,
                            'train_data': train_data,
                            'test_weeks': test_weeks
                        }
                        data_dicts[commodity] = data_dict
                        log(f"  ✓ Added data for {commodity} ({len(train_data)} train weeks)", log_callback)
                except Exception as e:
                    log(f"  ✗ Unexpected error for {commodity}: {str(e)}", log_callback)
            
            if not all_train_data:
                log(f"  ⚠ No valid data found for {data_type.upper()}. Skipping model generation.", log_callback)
                continue
                
            log(f"\nTraining unified GLOBAL {data_type.upper()} model...", log_callback)
            forecaster = LSTMForecaster(sequence_length=active_config['sequence_length'])
            
            try:
                history = forecaster.train_multiple(all_train_data, epochs=active_config['epochs'], verbose=1)
                log(f"✓ Global {data_type} model trained successfully.", log_callback)
            except Exception as e:
                log(f"❌ Failed to train global {data_type} model: {e}", log_callback)
                continue
            
            # Save the global model
            os.makedirs(active_config['model_dir'], exist_ok=True)
            global_model_path = f"{active_config['model_dir']}/global_{data_type}"
            forecaster.save_model(global_model_path)
            
            # Evaluate
            log(f"\nEvaluating global {data_type} model...", log_callback)
            for commodity, eval_data in test_data_dict.items():
                try:
                    pats = forecaster.forecast(eval_data['train_data'], weeks_ahead=len(eval_data['test_data']))
                    metrics = calculate_metrics(eval_data['test_data'], pats)
                    
                    results_summary.append({
                        'commodity': commodity,
                        'data_type': data_type,
                        'status': 'SUCCESS',
                        'accuracy': metrics['accuracy'],
                        'mae': metrics['mae'],
                        'weeks_trained': len(data_dicts[commodity]['values'])
                    })
                except Exception:
                    pass
                    
            # Save global metadata
            avg_acc = np.mean([r['accuracy'] for r in results_summary if r['status'] == 'SUCCESS'] or [0])
            metadata = {
                'model_type': 'Global',
                'data_type': data_type,
                'training_date': datetime.now().isoformat(),
                'model_config': {
                    'sequence_length': active_config['sequence_length'],
                    'epochs': active_config['epochs'],
                    'architecture': 'LSTM_Weekly_Global'
                },
                'performance_metrics': {
                    'accuracy': float(avg_acc),
                    'note': 'Average accuracy across all commodities'
                },
                'data_info': {
                    'commodities_included': [r['commodity'] for r in results_summary if r['status'] == 'SUCCESS']
                }
            }
            with open(f"{global_model_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            log(f"\n✓ Global Model saved: {global_model_path}_model.h5", log_callback)
            log(f"✓ Global Metadata saved: {global_model_path}_metadata.json", log_callback)
        
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
        
        # New Step: Archive redundant individual models
        log("\n" + "="*70, log_callback)
        log("CLEANING UP REDUNDANT MODELS", log_callback)
        log("="*70, log_callback)
        log("Archiving individual per-commodity models...", log_callback)
        archive_old_models(active_config['model_dir'])
        log("✓ Redundant models moved to 'models/archive/'.", log_callback)
        
        log("\nTraining process completed successfully.", log_callback)
        
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