import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, GRU, MultiHeadAttention, LayerNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from data_preprocessing import StockDataPreprocessor
from visualization import StockVisualizer
from config import (
    SEQUENCE_LENGTH, TRAIN_TEST_SPLIT, VALIDATION_SPLIT,
    LSTM_UNITS, LSTM_LAYERS, DROPOUT_RATE, LEARNING_RATE,
    BATCH_SIZE, EPOCHS, METRICS, DEFAULT_STOCK_SYMBOL
)
import os
import joblib
from datetime import datetime
import optuna
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from tensorflow.keras.regularizers import l2

class StockPredictor:
    def __init__(self):
        self.preprocessor = StockDataPreprocessor()
        self.visualizer = StockVisualizer()
        self.lstm_model = None
        self.gru_model = None
        self.history = None
        self.lr_schedule = None
        self.input_shape = None
        self.n_features = None
        self.model_dir = 'saved_models'
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_params = None
        self.scaler = MinMaxScaler()
        self.historical_data = None
    
    def build_transformer_model(self, input_shape):
        """Build a Transformer model for time series prediction"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention layer
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(attention + inputs)
        
        # Feed-forward network
        ffn = Dense(128, activation='relu')(x)
        ffn = Dense(input_shape[1])(ffn)
        x = LayerNormalization(epsilon=1e-6)(ffn + x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU model with improved architecture"""
        model = Sequential([
            # First GRU layer
            GRU(units=64, return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second GRU layer
            GRU(units=32, return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model with improved architecture"""
        model = Sequential([
            # First LSTM layer
            LSTM(units=64, return_sequences=True, 
                 input_shape=input_shape,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(units=32, return_sequences=False,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def build_hybrid_cnn_lstm_model(self, input_shape):
        """Build a hybrid CNN-LSTM model"""
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # LSTM layers for temporal dependencies
            LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(16, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        return model
    
    def save_models(self, symbol):
        """Save all trained models"""
        try:
            # Save LSTM model
            self.lstm_model.save(os.path.join(self.model_dir, f'{symbol}_lstm.h5'))
            
            # Save GRU model
            self.gru_model.save(os.path.join(self.model_dir, f'{symbol}_gru.h5'))
            
            # Save Hybrid CNN-LSTM model
            self.hybrid_model.save(os.path.join(self.model_dir, f'{symbol}_hybrid_cnn_lstm.h5'))
            
            # Save scaler
            with open(os.path.join(self.model_dir, f'{symbol}_scaler.pkl'), 'wb') as f:
                joblib.dump(self.scaler, f)
            
            print("All models saved successfully")
            return True
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, symbol):
        """Load all trained models"""
        try:
            # Check if model files exist first
            required_files = [
                f'{symbol}_lstm.h5',
                f'{symbol}_gru.h5',
                f'{symbol}_hybrid_cnn_lstm.h5',
                f'{symbol}_scaler.pkl'
            ]
            
            # Check if all required files exist
            for file in required_files:
                if not os.path.exists(os.path.join(self.model_dir, file)):
                    print(f"Model file {file} not found. Need to train models first.")
                    return False
            
            # If all files exist, load them
            self.lstm_model = tf.keras.models.load_model(os.path.join(self.model_dir, f'{symbol}_lstm.h5'))
            self.gru_model = tf.keras.models.load_model(os.path.join(self.model_dir, f'{symbol}_gru.h5'))
            self.hybrid_model = tf.keras.models.load_model(os.path.join(self.model_dir, f'{symbol}_hybrid_cnn_lstm.h5'))
            
            with open(os.path.join(self.model_dir, f'{symbol}_scaler.pkl'), 'rb') as f:
                self.scaler = joblib.load(f)
            
            print("All models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading models for {symbol}: {str(e)}")
            return False
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for hyperparameter optimization"""
        # Define the hyperparameter search space
        lstm_units = trial.suggest_int('lstm_units', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.001)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        
        # Build and train model with these parameters
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=self.input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Return validation loss
        return min(history.history['val_loss'])
    
    def tune_hyperparameters(self, X, y):
        """Perform hyperparameter tuning using Optuna"""
        try:
            # Split data for tuning
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create study object
            study = optuna.create_study(direction='minimize')
            
            # Run optimization
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=20,
                timeout=3600  # 1 hour timeout
            )
            
            # Store best parameters
            self.best_params = study.best_params
            print("Best hyperparameters:", self.best_params)
            
            return self.best_params
            
        except Exception as e:
            print(f"Error in hyperparameter tuning: {str(e)}")
            return None
    
    def check_stationarity(self, data):
        """Test for stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(data)
        return result[1] < 0.05  # Returns True if stationary (p-value < 0.05)

    def make_stationary(self, data):
        """Apply transformations to make the data stationary"""
        # Take log transform first to stabilize variance
        data_log = np.log1p(data)
        
        # Apply differencing until stationary
        diff_data = data_log.copy()
        d = 0
        while not self.check_stationarity(diff_data) and d < 2:
            diff_data = np.diff(diff_data)
            d += 1
        
        return diff_data, d, data_log

    def find_optimal_sarima_params(self, data):
        """Find optimal SARIMA parameters using auto_arima"""
        # Fit auto_arima model
        model = auto_arima(data,
                          start_p=0, start_q=0, max_p=3, max_q=3,
                          start_P=0, start_Q=0, max_P=2, max_Q=2,
                          m=12,  # Monthly seasonal pattern
                          seasonal=True,
                          d=None,  # Let model determine d
                          D=None,  # Let model determine D
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
        
        return model.order, model.seasonal_order

    def inverse_transform_sarima(self, predictions, d, data_log):
        """Inverse transform SARIMA predictions back to original scale"""
        # Integrate d times
        for _ in range(d):
            predictions = np.cumsum(predictions) + data_log[-1]
        
        # Inverse log transform
        return np.expm1(predictions)

    def prepare_sequences(self, df, sequence_length=SEQUENCE_LENGTH):
        """Prepare sequences with improved scaling"""
        try:
            # Select most important features
            feature_columns = [
                'close', 'volume',  # Basic price and volume
                'SMA_20', 'SMA_50',  # Moving averages
                'RSI',  # Momentum
                'BB_upper', 'BB_lower',  # Bollinger Bands
                'MACD',  # Trend
                'ATR',  # Volatility
                '%K', '%D'  # Stochastic
            ]
            
            # Validate all required columns exist
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise Exception(f"Missing columns: {missing_columns}")
            
            # Select features and handle missing values
            data = df[feature_columns].copy()
            data = data.ffill().bfill()
            
            # Check for any remaining NaN values
            if data.isna().any().any():
                raise Exception("Data still contains NaN values after filling")
            
            # Scale each feature independently
            scaled_data = np.zeros_like(data.values)
            for i, column in enumerate(data.columns):
                feature_scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled_data[:, i] = feature_scaler.fit_transform(data[[column]]).ravel()
                
                # Store the scaler for this feature
                if not hasattr(self, 'feature_scalers'):
                    self.feature_scalers = {}
                self.feature_scalers[column] = feature_scaler
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                # Use only the close price as target
                close_idx = feature_columns.index('close')
                y.append(scaled_data[i + sequence_length, close_idx])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error preparing sequences: {str(e)}")
            return None, None

    def train_models(self, symbol=DEFAULT_STOCK_SYMBOL, force_retrain=False, auto_tune=False):
        """Train all models"""
        if not force_retrain and os.path.exists(os.path.join(self.model_dir, f'{symbol}_lstm.h5')) and \
           os.path.exists(os.path.join(self.model_dir, f'{symbol}_gru.h5')) and \
           os.path.exists(os.path.join(self.model_dir, f'{symbol}_hybrid_cnn_lstm.h5')) and \
           os.path.exists(os.path.join(self.model_dir, f'{symbol}_scaler.pkl')):
            print("Loading existing models...")
            try:
                # Load models
                self.lstm_model = load_model(os.path.join(self.model_dir, f'{symbol}_lstm.h5'))
                self.gru_model = load_model(os.path.join(self.model_dir, f'{symbol}_gru.h5'))
                self.hybrid_model = load_model(os.path.join(self.model_dir, f'{symbol}_hybrid_cnn_lstm.h5'))
                
                # Load scaler
                with open(os.path.join(self.model_dir, f'{symbol}_scaler.pkl'), 'rb') as f:
                    self.scaler = joblib.load(f)
                
                print("Successfully loaded all models and scaler")
                
                # Get test data for evaluation
                X, y = self.preprocessor.prepare_data(symbol=symbol)
                if X is None or y is None:
                    print("Failed to prepare data")
                    return None, None
                
                # Store input shape for model building
                self.input_shape = (X.shape[1], X.shape[2])
                self.n_features = X.shape[2]
                
                # Split data into train and test sets
                split_idx = int(len(X) * TRAIN_TEST_SPLIT)
                X_test = X[split_idx:]
                y_test = y[split_idx:]
                
                return X_test, y_test
                
            except Exception as e:
                print(f"Error loading models: {str(e)}")
                print("Will retrain models...")
                force_retrain = True
        
        print("Training new models...")
        
        # Get and preprocess data
        X, y = self.preprocessor.prepare_data(symbol=symbol)
        if X is None or y is None:
            print("Failed to prepare data")
            return None, None
        
        # Store input shape for model building
        self.input_shape = (X.shape[1], X.shape[2])
        self.n_features = X.shape[2]
        
        # Split data into train and test sets
        split_idx = int(len(X) * TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Hyperparameter tuning if enabled
        if auto_tune:
            print("Performing hyperparameter tuning...")
            best_params = self.tune_hyperparameters(X_train, y_train)
            if best_params:
                global LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE
                LSTM_UNITS = best_params['lstm_units']
                DROPOUT_RATE = best_params['dropout_rate']
                LEARNING_RATE = best_params['learning_rate']
                BATCH_SIZE = best_params['batch_size']
        
        # Build and compile models
        print("Building models...")
        self.lstm_model = self.build_lstm_model(self.input_shape)
        self.gru_model = self.build_gru_model(self.input_shape)
        self.hybrid_model = self.build_hybrid_cnn_lstm_model(self.input_shape)
        
        # Define callbacks with early stopping and learning rate reduction
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train LSTM model
        print("\nTraining LSTM model...")
        self.lstm_history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_train, y_train),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train GRU model
        print("\nTraining GRU model...")
        self.gru_history = self.gru_model.fit(
            X_train, y_train,
            validation_data=(X_train, y_train),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train Hybrid CNN-LSTM model
        print("\nTraining Hybrid CNN-LSTM model...")
        self.hybrid_history = self.hybrid_model.fit(
            X_train, y_train,
            validation_data=(X_train, y_train),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save all models
        self.save_models(symbol)
        
        return X_test, y_test
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate models and calculate ensemble weights based on performance.
        """
        metrics = {}
        predictions = {}
        
        # Get predictions from each model
        try:
            lstm_pred = self.lstm_model.predict(X_test).reshape(-1)  # Reshape to 1D array
            predictions['lstm'] = lstm_pred
            metrics['lstm'] = {
                'mae': mean_absolute_error(y_test, lstm_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, lstm_pred)),
                'r2': r2_score(y_test, lstm_pred)
            }
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            metrics['lstm'] = {'mae': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}
        
        try:
            gru_pred = self.gru_model.predict(X_test).reshape(-1)  # Reshape to 1D array
            predictions['gru'] = gru_pred
            metrics['gru'] = {
                'mae': mean_absolute_error(y_test, gru_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, gru_pred)),
                'r2': r2_score(y_test, gru_pred)
            }
        except Exception as e:
            print(f"Error in GRU prediction: {e}")
            metrics['gru'] = {'mae': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}
        
        try:
            hybrid_pred = self.hybrid_model.predict(X_test).reshape(-1)  # Reshape to 1D array
            predictions['hybrid'] = hybrid_pred
            metrics['hybrid'] = {
                'mae': mean_absolute_error(y_test, hybrid_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, hybrid_pred)),
                'r2': r2_score(y_test, hybrid_pred)
            }
        except Exception as e:
            print(f"Error in Hybrid CNN-LSTM prediction: {e}")
            metrics['hybrid'] = {'mae': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}
        
        # Calculate weights based on MAE and R²
        weights = {}
        mae_scores = {model: 1/metrics[model]['mae'] if metrics[model]['mae'] != float('inf') else 0 
                     for model in metrics}
        r2_scores = {model: max(0, metrics[model]['r2']) if metrics[model]['r2'] != float('-inf') else 0 
                    for model in metrics}
        
        # Normalize scores
        total_mae = sum(mae_scores.values())
        total_r2 = sum(r2_scores.values())
        
        if total_mae > 0 and total_r2 > 0:
            mae_weights = {model: score/total_mae for model, score in mae_scores.items()}
            r2_weights = {model: score/total_r2 for model, score in r2_scores.items()}
            
            # Combine weights (50% MAE, 50% R²)
            weights = {model: 0.5 * mae_weights[model] + 0.5 * r2_weights[model] 
                      for model in metrics}
            
            # Normalize final weights
            total_weight = sum(weights.values())
            weights = {model: weight/total_weight for model, weight in weights.items()}
        else:
            # Default weights if metrics are invalid
            weights = {'lstm': 0.4, 'gru': 0.3, 'hybrid': 0.3}
        
        # Store weights for future predictions
        self.model_weights = weights
        
        print("\nEnsemble Weights:")
        for model, weight in weights.items():
            print(f"{model.upper()}: {weight:.2%}")
        
        # Calculate ensemble predictions
        ensemble_pred = np.zeros_like(y_test.reshape(-1))  # Ensure 1D array
        for model, weight in weights.items():
            if model in predictions and weight > 0:
                ensemble_pred += weight * predictions[model]
        
        metrics['ensemble'] = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'r2': r2_score(y_test, ensemble_pred)
        }
        
        print("\nModel Performance:")
        for model in metrics:
            print(f"\n{model.upper()}:")
            print(f"MAE: {metrics[model]['mae']:.4f}")
            print(f"RMSE: {metrics[model]['rmse']:.4f}")
            print(f"R²: {metrics[model]['r2']:.4f}")
        
        return metrics
    
    def plot_results(self, metrics, X_test, y_test):
        """
        Plot the model results and save visualizations.
        """
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Model Performance Comparison
        plt.subplot(2, 1, 1)
        models = ['LSTM', 'GRU', 'Hybrid CNN-LSTM']
        rmse_scores = [metrics[m.lower()]['rmse'] for m in models]
        mae_scores = [metrics[m.lower()]['mae'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, rmse_scores, width, label='RMSE', color='#2ecc71', alpha=0.7)
        plt.bar(x + width/2, mae_scores, width, label='MAE', color='#3498db', alpha=0.7)
        plt.xlabel('Models', fontsize=10)
        plt.ylabel('Error Metrics', fontsize=10)
        plt.title('Model Performance Comparison', fontsize=12, pad=15)
        plt.xticks(x, models, fontsize=10)
        plt.legend(fontsize=10)
        
        # Plot 2: Actual vs Predicted Values
        plt.subplot(2, 1, 2)
        
        # Get predictions from each model
        lstm_pred = self.lstm_model.predict(X_test).ravel()
        gru_pred = self.gru_model.predict(X_test).ravel()
        hybrid_pred = self.hybrid_model.predict(X_test).ravel()
        
        # Calculate ensemble predictions using stored weights
        weights = self.model_weights if hasattr(self, 'model_weights') else {'lstm': 0.4, 'gru': 0.3, 'hybrid': 0.3}
        ensemble_pred = lstm_pred * weights['lstm'] + gru_pred * weights['gru'] + hybrid_pred * weights['hybrid']
        
        plt.plot(y_test, label='Actual', color='#2c3e50', linewidth=2, alpha=0.8)
        plt.plot(ensemble_pred, label='Ensemble', color='#e74c3c', linewidth=2, alpha=0.8)
        plt.plot(lstm_pred, label='LSTM', color='#2ecc71', linewidth=1, alpha=0.4)
        plt.plot(gru_pred, label='GRU', color='#3498db', linewidth=1, alpha=0.4)
        plt.plot(hybrid_pred, label='Hybrid CNN-LSTM', color='#9b59b6', linewidth=1, alpha=0.4)
        
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Stock Price', fontsize=10)
        plt.title('Actual vs Predicted Values', fontsize=12, pad=15)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✅ Visualizations saved as 'model_results.png'")
    
    def predict_next_day(self, symbol):
        """
        Predict the next day's price using the ensemble of models.
        """
        try:
            # Get and preprocess the data using the same pipeline as training
            data = self.preprocessor.get_historical_data(symbol)
            if data is None or data.empty:
                raise ValueError("Failed to get historical data")
            
            # Get the last actual close price for reference
            last_close = float(data['close'].iloc[-1])
            print(f"Last close price: ${last_close:.2f}")
            
            # Prepare features using the same preprocessing pipeline
            features = self.preprocessor.add_technical_indicators(data.copy())
            if features is None:
                raise ValueError("Failed to prepare features")
            
            # Scale the features using the same scaler from training
            scaled_features = self.preprocessor.scale_features(features)
            if scaled_features is None:
                raise ValueError("Failed to scale features")
            
            # Create sequences using the same method as in training
            X = self.preprocessor.create_sequences(scaled_features)
            if X is None:
                raise ValueError("Failed to create sequences")
            
            # Take only the last sequence for prediction
            X = X[-1:].astype(np.float32)
            
            # Get predictions from each model
            predictions = {}
            weights = self.model_weights if hasattr(self, 'model_weights') else {'lstm': 0.4, 'gru': 0.3, 'hybrid': 0.3}
            
            # Get LSTM prediction
            if weights.get('lstm', 0) > 0:
                lstm_pred = float(self.lstm_model.predict(X, verbose=0)[0][0])
                # Inverse transform the prediction to get actual price
                lstm_price = self.preprocessor.inverse_transform_price(lstm_pred)
                predictions['lstm'] = lstm_price
                print(f"LSTM prediction: ${lstm_price:.2f}")
            
            # Get GRU prediction
            if weights.get('gru', 0) > 0:
                gru_pred = float(self.gru_model.predict(X, verbose=0)[0][0])
                # Inverse transform the prediction to get actual price
                gru_price = self.preprocessor.inverse_transform_price(gru_pred)
                predictions['gru'] =  gru_price
                print(f"GRU prediction: ${gru_price:.2f}")
            
            # Get Hybrid CNN-LSTM prediction
            if weights.get('hybrid', 0) > 0:
                hybrid_pred = float(self.hybrid_model.predict(X, verbose=0)[0][0])
                # Inverse transform the prediction to get actual price
                hybrid_price = self.preprocessor.inverse_transform_price(hybrid_pred)
                predictions['hybrid'] = hybrid_price
                print(f"Hybrid CNN-LSTM prediction: ${hybrid_price:.2f}")
            
            # Calculate weighted ensemble prediction
            ensemble_pred = 0.0
            valid_weight_sum = 0.0
            
            for model, pred in predictions.items():
                weight = weights.get(model, 0)
                if weight > 0:
                    ensemble_pred += weight * pred
                    valid_weight_sum += weight
            
            if valid_weight_sum > 0:
                # Normalize the ensemble prediction
                predicted_price = ensemble_pred / valid_weight_sum
                
                # Add some randomness based on historical volatility
                if hasattr(self, 'historical_data') and not self.historical_data.empty:
                    # Calculate daily volatility from the last 30 days
                    volatility = self.historical_data['close'].pct_change().tail(30).std()
                    # Add random variation within 1 standard deviation
                    random_factor = 1 + np.random.normal(0, volatility)
                    predicted_price *= random_factor
                
                print(f"\nEnsemble Prediction:")
                print(f"Last closing price: ${last_close:.2f}")
                print(f"Predicted price change: ${predicted_price - last_close:.2f}")
                print(f"Predicted price for next day: ${predicted_price:.2f}")
                
                # Update historical data with the new prediction
                if hasattr(self, 'historical_data'):
                    # Create timezone-naive datetime for the next day
                    next_day = pd.Timestamp(self.historical_data.index[-1] + pd.Timedelta(days=1)).tz_localize(None)
                    new_row = pd.DataFrame({
                        'close': [predicted_price],
                        'open': [last_close],  # Use last close as next day's open
                        'high': [max(predicted_price, last_close)],
                        'low': [min(predicted_price, last_close)],
                        'volume': [self.historical_data['volume'].mean()]  # Use average volume
                    }, index=[next_day])
                    
                    # Ensure historical data index is timezone-naive
                    if self.historical_data.index.tz is not None:
                        self.historical_data.index = self.historical_data.index.tz_localize(None)
                    
                    self.historical_data = pd.concat([self.historical_data, new_row])
                
                return predicted_price
            else:
                raise ValueError("No valid predictions available")
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

def main():
    # Initialize predictor
    predictor = StockPredictor()
    
    # Train models
    print("Training models...")
    X_test, y_test = predictor.train_models(force_retrain=True)
    
    if X_test is not None:
        # Evaluate models
        print("\nEvaluating models...")
        metrics = predictor.evaluate_models(X_test, y_test)
        
        # Plot results
        print("\nGenerating visualizations...")
        predictor.plot_results(metrics, X_test, y_test)
        
        # Predict next day
        print("\nPredicting next day's price...")
        next_day_pred = predictor.predict_next_day(DEFAULT_STOCK_SYMBOL)
        if next_day_pred is not None:
            print(f"Predicted price for next day: ${next_day_pred:.2f}")
        else:
            print("Failed to predict next day's price.")

if __name__ == "__main__":
    main() 