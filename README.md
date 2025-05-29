# Advanced Stock Price Predictor

An advanced stock price prediction system that leverages an ensemble of deep learning models to forecast future stock prices. The system combines three sophisticated neural network architectures:

1. **LSTM (Long Short-Term Memory)**: Specializes in capturing long-term dependencies in time series data
2. **GRU (Gated Recurrent Unit)**: Provides efficient sequence modeling with reduced complexity
3. **Hybrid CNN-LSTM**: Combines convolutional layers for pattern recognition with LSTM layers for temporal analysis

## Features

- Real-time stock data fetching using yfinance
- Comprehensive technical indicators:
  - Moving Averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - ATR (Average True Range)
  - ROC (Rate of Change)
  - OBV (On-Balance Volume)
  - Momentum
  - Volatility metrics
- Dynamic model weight adjustment based on MAE and R² scores
- Interactive Streamlit UI with:
  - Model performance metrics
  - Training history visualization
  - Future price predictions
  - Technical analysis charts
- Advanced preprocessing with separate scaling for:
  - Price-related features (open, high, low, close, volume)
  - Technical indicators
- Support for 20+ major stock symbols
- Model persistence with automatic loading/saving
- Early stopping and learning rate reduction for better training
- Hyperparameter optimization using Optuna

## Model Architecture

### LSTM Model
- Two LSTM layers (64 and 32 units)
- L2 regularization (0.01)
- Batch normalization after each layer
- Dropout (0.2) after LSTM layers, (0.1) after dense layer
- Dense layers (16 units with ReLU, 1 unit for prediction)
- Adam optimizer (learning rate: 0.001)

### GRU Model
- Two GRU layers (64 and 32 units)
- L2 regularization (0.01)
- Batch normalization after each layer
- Dropout (0.2) after GRU layers, (0.1) after dense layer
- Dense layers (16 units with ReLU, 1 unit for prediction)
- Adam optimizer (learning rate: 0.001)

### Hybrid CNN-LSTM Model
- CNN layer (32 filters, kernel size 3)
- MaxPooling1D (pool size 2)
- Two LSTM layers (32 and 16 units)
- L2 regularization (0.01)
- Batch normalization after each layer
- Dropout (0.2) after LSTM layers, (0.1) after dense layer
- Dense layers (16 units with ReLU, 1 unit for prediction)
- Adam optimizer (learning rate: 0.001)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aayush-05/Voila.git
cd Voila
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Select a stock symbol from the dropdown menu
3. Choose training parameters:
   - Training start date
   - Prediction end date
   - Force retrain model (optional)
   - Auto-tune hyperparameters (optional)
   - Training data years
   - LSTM/GRU units (32-256)
   - Dropout rate (0.1-0.5)
   - Learning rate (0.0001-0.01)
   - Batch size (16-128)

4. Click "Train Model and Predict" to start the process

## Project Structure

```
stockprediction/
├── app.py                 # Main Streamlit application
├── model.py              # Model architecture and training logic
├── data_preprocessing.py # Data fetching and preprocessing
├── visualization.py      # Plotting and visualization functions
├── config.py            # Configuration parameters
├── requirements.txt     # Project dependencies
└── saved_models/       # Directory for saved model files
```

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- yfinance
- pandas
- numpy
- scikit-learn
- plotly
- ta (Technical Analysis library)
- optuna (for hyperparameter optimization)
- joblib (for model persistence)

## Acknowledgments

- yfinance for providing stock data
- TensorFlow team for the deep learning framework
- Streamlit team for the web framework
- Technical Analysis library (ta) for indicators 
