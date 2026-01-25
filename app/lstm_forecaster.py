import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import joblib

class LSTMForecaster:
    def __init__(self, sequence_length=4):
        """
        Initialize LSTM Forecaster for weekly data
        
        Args:
            sequence_length: Number of past weeks to use for prediction (default: 4 weeks)
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model optimized for weekly agricultural data
        
        Architecture:
        - 2 LSTM layers with dropout for regularization
        - Dense layers for final prediction
        - Designed for weekly time series forecasting
        """
        model = keras.Sequential([
            # First LSTM layer with return sequences
            keras.layers.LSTM(
                50, 
                activation='relu', 
                return_sequences=True, 
                input_shape=input_shape
            ),
            keras.layers.Dropout(0.2),
            
            # Second LSTM layer
            keras.layers.LSTM(50, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Dense layers
            keras.layers.Dense(25, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam', 
            loss='mse', 
            metrics=['mae']
        )
        return model
    
    def train(self, data, epochs=50, verbose=0):
        """
        Train the LSTM model on weekly data
        
        Args:
            data: Array of historical values (weekly data points)
            epochs: Number of training epochs
            verbose: Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
            Training history
        """
        # Normalize data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Validation: Need at least 2 sequences for training
        # With sequence_length=4, need minimum 6 weeks (4 for sequence + 2 for training)
        min_sequences = 2
        if len(X) < min_sequences:
            raise ValueError(
                f"Insufficient data for training. Need at least {self.sequence_length + min_sequences} weeks. "
                f"Found: {len(data)} weeks, which creates {len(X)} training sequences."
            )
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self.build_model((X.shape[1], 1))
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=max(1, len(X) // 4),  # Dynamic batch size
            verbose=verbose,
            validation_split=0.1 if len(X) >= 10 else 0,  # Only use validation if enough data
            callbacks=[early_stop]
        )
        
        return history
    
    def forecast(self, data, weeks_ahead):
        """
        Generate forecasts for future weeks
        
        Args:
            data: Historical data array (weekly values)
            weeks_ahead: Number of weeks to forecast into the future
        
        Returns:
            Array of forecasted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} weeks of data for forecasting. "
                f"Provided: {len(data)} weeks"
            )
        
        # Normalize data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        # Use last sequence_length weeks as initial sequence
        current_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        forecasts = []
        for week in range(weeks_ahead):
            # Predict next week's value
            next_pred = self.model.predict(current_sequence, verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update sequence: remove oldest week, add new prediction
            current_sequence = np.append(
                current_sequence[:, 1:, :], 
                next_pred.reshape(1, 1, 1), 
                axis=1
            )
        
        # Inverse transform to original scale
        forecasts = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
        return forecasts.flatten()
    
    def predict_next_week(self, data):
        """
        Predict only the next week's value
        
        Args:
            data: Historical weekly data
            
        Returns:
            Single predicted value for next week
        """
        forecast = self.forecast(data, weeks_ahead=1)
        return forecast[0]
    
    def save_model(self, filepath):
        """
        Save trained model and scaler
        
        Args:
            filepath: Base path for saving (will create _model.h5 and _scaler.pkl)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(f"{filepath}_model.h5")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'model_type': 'LSTM_Weekly'
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
    
    def load_model(self, filepath):
        """
        Load trained model and scaler
        
        Args:
            filepath: Base path for loading
        """
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        # Load metadata if available
        try:
            metadata = joblib.load(f"{filepath}_metadata.pkl")
            self.sequence_length = metadata.get('sequence_length', self.sequence_length)
        except:
            pass


def calculate_metrics(actual, predicted):
    """
    Calculate forecasting performance metrics
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
    
    Returns:
        Dictionary with MSE, RMSE, MAE, and MAPE metrics
    """
    # Ensure arrays have same length
    min_length = min(len(actual), len(predicted))
    actual = actual[:min_length]
    predicted = predicted[:min_length]
    
    # Calculate metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # MAPE: avoid division by zero
    mape_values = []
    for a, p in zip(actual, predicted):
        if a != 0:
            mape_values.append(np.abs((a - p) / a))
    
    mape = np.mean(mape_values) * 100 if mape_values else 0
    
    # Calculate accuracy percentage (100 - MAPE)
    accuracy = max(0, 100 - mape)
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "accuracy": float(accuracy)
    }


def evaluate_model(forecaster, train_data, test_data, weeks_ahead):
    """
    Evaluate model performance on test data
    
    Args:
        forecaster: Trained LSTMForecaster instance
        train_data: Training data array
        test_data: Test data array
        weeks_ahead: Number of weeks to forecast
    
    Returns:
        Dictionary with predictions and metrics
    """
    # Generate predictions
    predictions = forecaster.forecast(train_data, weeks_ahead)
    
    # Calculate metrics
    metrics = calculate_metrics(test_data[:weeks_ahead], predictions)
    
    return {
        "predictions": predictions.tolist(),
        "actual": test_data[:weeks_ahead].tolist(),
        "metrics": metrics
    }


def cross_validate_forecast(data, sequence_length=4, n_splits=3):
    """
    Perform time series cross-validation
    
    Args:
        data: Complete dataset
        sequence_length: Sequence length for LSTM
        n_splits: Number of validation splits
    
    Returns:
        Average metrics across all splits
    """
    all_metrics = []
    split_size = len(data) // (n_splits + 1)
    
    for i in range(n_splits):
        # Split data
        train_end = split_size * (i + 1)
        train_data = data[:train_end]
        test_data = data[train_end:train_end + 4]  # Test on next 4 weeks
        
        if len(train_data) < sequence_length + 2 or len(test_data) < 1:
            continue
        
        # Train and evaluate
        forecaster = LSTMForecaster(sequence_length=sequence_length)
        forecaster.train(train_data, epochs=30, verbose=0)
        
        predictions = forecaster.forecast(train_data, weeks_ahead=len(test_data))
        metrics = calculate_metrics(test_data, predictions)
        all_metrics.append(metrics)
    
    if not all_metrics:
        return None
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return avg_metrics