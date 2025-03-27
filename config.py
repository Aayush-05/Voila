import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# API Configuration
MARKET_STACK_API_KEY = os.getenv('MARKETSTACK_API_KEY')
MARKET_STACK_BASE_URL = "http://api.marketstack.com/v1"
RATE_LIMIT_PAUSE = 15  # seconds to wait between API calls
MAX_RETRIES = 3

# Data Configuration
DEFAULT_STOCK_SYMBOL = 'AAPL'  # Default stock to analyze
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Last 1 year of data

# Model Configuration
SEQUENCE_LENGTH = 60          # Number of time steps to look back
TRAIN_TEST_SPLIT = 0.8       # Ratio of training to test data
VALIDATION_SPLIT = 0.2       # Ratio of training data to use for validation

# LSTM Model Parameters
LSTM_UNITS = 50
LSTM_LAYERS = 2
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# XGBoost Parameters
XGB_LEARNING_RATE = 0.001
XGB_MAX_DEPTH = 5
XGB_N_ESTIMATORS = 100

# Technical Indicators Parameters
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Feature Engineering
USE_TECHNICAL_INDICATORS = True
USE_PRICE_CHANGES = True
USE_VOLUME_CHANGES = True
USE_MARKET_SENTIMENT = True

# Visualization Settings
PLOT_THEME = "plotly_white"
PLOT_HEIGHT = 600
PLOT_WIDTH = 1000

# Model Evaluation
METRICS = ['rmse', 'mae', 'r2']

# API Endpoints
MARKET_STACK_ENDPOINTS = {
    'eod': '/eod',           # End of day data
    'intraday': '/intraday', # Intraday data
    'tickers': '/tickers',   # Stock symbols
    'exchanges': '/exchanges' # Stock exchanges
} 