import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_technical_indicators(df):
    """Calculate additional technical indicators"""
    # ATR (Average True Range)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['atr'] = tr.rolling(window=14).mean()
    
    # Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    
    df['stoch_k'] = 100 * ((close - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    
    return df

def create_time_features(df):
    """Create time-based features"""
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Create cyclical features
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def calculate_price_momentum(df):
    """Calculate price momentum indicators"""
    # Price momentum
    df['momentum'] = df['close'].pct_change(periods=10)
    
    # Rate of change
    df['roc'] = df['close'].pct_change(periods=12)
    
    # Price acceleration
    df['acceleration'] = df['momentum'].diff()
    
    return df

def add_market_regime_features(df):
    """Add market regime features"""
    # Volatility regime
    df['volatility_regime'] = df['close'].pct_change().rolling(window=20).std()
    
    # Trend strength
    df['trend_strength'] = abs(df['close'].pct_change(periods=20))
    
    # Market regime (1: uptrend, -1: downtrend, 0: sideways)
    df['market_regime'] = 0
    df.loc[df['close'] > df['close'].rolling(window=20).mean(), 'market_regime'] = 1
    df.loc[df['close'] < df['close'].rolling(window=20).mean(), 'market_regime'] = -1
    
    return df

def prepare_features(df):
    """Prepare all features for the model"""
    df = calculate_technical_indicators(df)
    df = create_time_features(df)
    df = calculate_price_momentum(df)
    df = add_market_regime_features(df)
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def get_trading_days(start_date, end_date):
    """Get number of trading days between two dates"""
    # Exclude weekends
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
    return len(trading_days)

def calculate_position_size(account_value, risk_per_trade, stop_loss_pct):
    """Calculate position size based on risk management rules"""
    risk_amount = account_value * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return position_size 