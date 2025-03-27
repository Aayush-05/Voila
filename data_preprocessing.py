import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler
from config import *
import time

class StockDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.historical_data = None
    
    def fetch_stock_data(self, symbol, days=None):
        """Fetch historical stock data using yfinance"""
        try:
            # Calculate dates
            end_date = datetime.now()
            if days:
                start_date = end_date - timedelta(days=days)
            else:
                # Get at least 2 years of data for better training
                start_date = end_date - timedelta(days=730)
                end_date = datetime.now()

            print(f"\nFetching data for {symbol}...")
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Ensure symbol is a string
            if isinstance(symbol, np.ndarray):
                symbol = str(symbol.item())
            elif not isinstance(symbol, str):
                symbol = str(symbol)
            
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            if df.empty:
                print(f"❌ No data found for {symbol}")
                return None
                
            # Print data info for debugging
            print("\nData Info:")
            print(f"Shape: {df.shape}")
            print("\nColumns:", df.columns.tolist())
            print("\nFirst few rows:")
            print(df.head())
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                print("\nMissing values:")
                print(missing_values[missing_values > 0])
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            print(f"\n✅ Successfully fetched {len(df)} days of data")
            return df
            
        except Exception as e:
            print(f"\n❌ Error fetching stock data: {str(e)}")
            print("Full error:")
            import traceback
            print(traceback.format_exc())
            return None
    
    def get_latest_data(self, symbol):
        """Get the latest data for prediction"""
        try:
            # Fetch recent data (SEQUENCE_LENGTH + 10 days of data)
            df = self.fetch_stock_data(
                symbol=symbol,
                days=SEQUENCE_LENGTH + 10  # Extra days for safety
            )
            
            if df is None or len(df) < SEQUENCE_LENGTH:
                raise Exception("Not enough recent data available for prediction")
            
            return df
        except Exception as e:
            print(f"Error getting latest data: {str(e)}")
            return None
    
    def get_historical_data(self, symbol):
        """Get historical data for a given symbol"""
        try:
            # Fetch data from Market Stack API
            data = self.fetch_stock_data(symbol)
            if data is not None and not data.empty:
                self.historical_data = data
                return data
            return None
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            return None
    
    def get_feature_columns(self):
        """Get list of feature columns for the model"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_20', 'SMA_50', 'RSI', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
            '%K', '%D', 'ATR', 'ROC', 'OBV', 'Momentum',
            'price_change', 'volume_change', 'volatility', 'daily_range'
        ]
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe."""
        try:
            if df is None or len(df) == 0:
                print("Error: Invalid dataframe for technical indicators")
                return None
            
            # Moving Averages
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # RSI
            df['RSI'] = self.calculate_rsi(df['close'])
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            # Stochastic Oscillator
            df['14-high'] = df['high'].rolling(14).max()
            df['14-low'] = df['low'].rolling(14).min()
            df['%K'] = (df['close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
            df['%D'] = df['%K'].rolling(3).mean()
            
            # Average True Range (ATR)
            df['TR'] = pd.DataFrame({
                'HL': df['high'] - df['low'],
                'HC': abs(df['high'] - df['close'].shift(1)),
                'LC': abs(df['low'] - df['close'].shift(1))
            }).max(axis=1)
            df['ATR'] = df['TR'].rolling(14).mean()
            
            # Price Rate of Change
            df['ROC'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # On-Balance Volume (OBV)
            df['OBV'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
            
            # Momentum
            df['Momentum'] = df['close'] - df['close'].shift(4)
            
            # Additional Features
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['volatility'] = df['close'].rolling(window=20).std()
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            
            # Handle missing values using newer methods
            df = df.ffill()
            df = df.bfill()
            
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return None
    
    def create_features(self, df):
        """Create additional features"""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Volatility
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # High-Low range
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def prepare_sequences(self, df, sequence_length=SEQUENCE_LENGTH):
        """Prepare sequences for LSTM model"""
        try:
            # Get all feature columns
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_20', 'SMA_50', 'RSI', 'EMA_12', 'EMA_26',
                'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
                '%K', '%D', 'ATR', 'ROC', 'OBV', 'Momentum',
                'price_change', 'volume_change', 'volatility', 'daily_range'
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
            
            # Scale features
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                y.append(scaled_data[i + sequence_length, data.columns.get_loc('close')])  # Get close price index
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error preparing sequences: {str(e)}")
            return None, None
    
    def prepare_data(self, symbol):
        """Prepare data for training"""
        try:
            # Get historical data
            df = self.get_historical_data(symbol)
            if df is None:
                return None, None
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            if df is None:
                return None, None
            
            # Create sequences
            X, y = self.prepare_sequences(df)
            if X is None or y is None:
                return None, None
            
            # Scale features separately
            # Price-related features
            price_features = ['open', 'high', 'low', 'close', 'volume']
            price_scaler = MinMaxScaler()
            X[:, :, :5] = price_scaler.fit_transform(X[:, :, :5].reshape(-1, 5)).reshape(X[:, :, :5].shape)
            
            # Technical indicators
            tech_features = X[:, :, 5:]
            tech_scaler = MinMaxScaler()
            X[:, :, 5:] = tech_scaler.fit_transform(tech_features.reshape(-1, tech_features.shape[-1])).reshape(tech_features.shape)
            
            # Scale target values
            y_scaler = MinMaxScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1))
            
            # Store scalers for later use
            self.price_scaler = price_scaler
            self.tech_scaler = tech_scaler
            self.y_scaler = y_scaler
            
            return X, y
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None
    
    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original scale"""
        dummy_array = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy_array[:, 3] = predictions  # 3 is the index of 'close' price
        return self.scaler.inverse_transform(dummy_array)[:, 3]
    
    def prepare_prediction_data(self, data):
        """Prepare data for prediction"""
        try:
            if data is None or len(data) < SEQUENCE_LENGTH:
                print(f"Error: Not enough data points. Got {len(data) if data is not None else 0}, need {SEQUENCE_LENGTH}")
                return None
            
            # Add technical indicators
            data = self.add_technical_indicators(data.copy())
            if data is None:
                print("Error: Failed to calculate technical indicators")
                return None
            
            # Get feature columns
            feature_columns = self.get_feature_columns()
            
            # Validate all required columns exist
            missing_columns = [col for col in feature_columns if col not in data.columns]
            if missing_columns:
                print(f"Error: Missing columns: {missing_columns}")
                return None
            
            # Select features
            data = data[feature_columns].copy()
            
            # Handle missing values
            data = data.ffill()
            data = data.bfill()
            
            # Check for any remaining NaN values
            if data.isna().any().any():
                print("Error: Data still contains NaN values after filling")
                return None
            
            # Scale the features
            try:
                scaled_data = self.scaler.fit_transform(data)
            except Exception as e:
                print(f"Error scaling data: {str(e)}")
                return None
            
            # Create sequence
            sequence = scaled_data[-SEQUENCE_LENGTH:]
            if len(sequence) != SEQUENCE_LENGTH:
                print(f"Error: Sequence length mismatch. Expected {SEQUENCE_LENGTH}, got {len(sequence)}")
                return None
            
            # Reshape for model input (batch_size, sequence_length, n_features)
            return np.array([sequence])
            
        except Exception as e:
            print(f"Error preparing prediction data: {str(e)}")
            return None
    
    def create_sequences(self, data):
        """
        Create sequences from the feature data for model input.
        
        Args:
            data (pd.DataFrame): Feature data with technical indicators
            
        Returns:
            np.ndarray: Sequences of shape (n_samples, sequence_length, n_features)
        """
        try:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            # Drop any columns with NaN values
            numeric_data = numeric_data.dropna(axis=1)
            
            # Convert to numpy array
            values = numeric_data.values
            
            # Create sequences
            sequences = []
            for i in range(len(values) - SEQUENCE_LENGTH):
                sequence = values[i:(i + SEQUENCE_LENGTH)]
                sequences.append(sequence)
            
            # Convert to numpy array
            sequences = np.array(sequences)
            
            # Take only the first 27 features if we have more
            if sequences.shape[2] > 27:
                sequences = sequences[:, :, :27]
            
            return sequences.astype(np.float32)
            
        except Exception as e:
            print(f"Error creating sequences: {str(e)}")
            return None
    
    def scale_features(self, data):
        """Scale features using MinMaxScaler"""
        try:
            # Get feature columns
            feature_columns = self.get_feature_columns()
            
            # Validate data
            if data is None or len(data) == 0:
                raise ValueError("Invalid data for scaling")
            
            # Scale features
            scaled_data = data.copy()
            scaled_data[feature_columns] = self.scaler.fit_transform(data[feature_columns])
            
            return scaled_data
        except Exception as e:
            print(f"Error scaling features: {str(e)}")
            return None
    
    def inverse_transform_price(self, scaled_price):
        """Inverse transform a single price prediction"""
        try:
            # Get the column index for 'close' price in feature columns
            feature_columns = self.get_feature_columns()
            close_idx = feature_columns.index('close')
            
            # Create a dummy array with zeros except for the predicted price
            dummy = np.zeros((1, len(feature_columns)))
            dummy[0, close_idx] = scaled_price
            
            # Inverse transform
            original_dummy = self.scaler.inverse_transform(dummy)
            return original_dummy[0, close_idx]
        except Exception as e:
            print(f"Error inverse transforming price: {str(e)}")
            return None 