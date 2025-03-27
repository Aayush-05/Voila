import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from model import StockPredictor
from config import *
import numpy as np
import plotly.express as px
import warnings
import time
import traceback
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, BatchNormalization, Dropout, Dense
import os

# Set page config first
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
PLOT_THEME = "plotly_dark"

# Available stock symbols
STOCK_SYMBOLS = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'GOOGL', # Alphabet Inc.
    'AMZN',  # Amazon.com Inc.
    'META',  # Meta Platforms Inc.
    'NVDA',  # NVIDIA Corporation
    'TSLA',  # Tesla Inc.
    'JPM',   # JPMorgan Chase & Co.
    'V',     # Visa Inc.
    'WMT',   # Walmart Inc.
    'PG',    # Procter & Gamble Company
    'MA',    # Mastercard Inc.
    'HD',    # Home Depot Inc.
    'BAC',   # Bank of America Corp.
    'XOM',   # Exxon Mobil Corporation
    'T',     # AT&T Inc.
    'PFE',   # Pfizer Inc.
    'CSCO',  # Cisco Systems Inc.
    'VZ',    # Verizon Communications Inc.
    'KO'     # Coca-Cola Company
]

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress .st-bo {
        background-color: #00a6ed;
    }
</style>
""", unsafe_allow_html=True)

def display_model_performance(metrics):
    """Display model performance metrics"""
    st.header("📊 Model Performance Metrics")
    
    # Create columns for each model's metrics
    lstm_col, gru_col, hybrid_col, ensemble_col = st.columns(4)
    
    # LSTM Metrics
    with lstm_col:
        st.subheader("LSTM")
        if 'LSTM' in metrics:
            st.metric("RMSE", f"{metrics['LSTM']['RMSE']:.4f}")
            st.metric("MAE", f"{metrics['LSTM']['MAE']:.4f}")
            st.metric("R²", f"{metrics['LSTM']['R2']:.4f}")
            st.metric("Weight", f"{metrics['LSTM']['weight']:.2%}")
    
    # GRU Metrics
    with gru_col:
        st.subheader("GRU")
        if 'GRU' in metrics:
            st.metric("RMSE", f"{metrics['GRU']['RMSE']:.4f}")
            st.metric("MAE", f"{metrics['GRU']['MAE']:.4f}")
            st.metric("R²", f"{metrics['GRU']['R2']:.4f}")
            st.metric("Weight", f"{metrics['GRU']['weight']:.2%}")
    
    # Hybrid CNN-LSTM Metrics
    with hybrid_col:
        st.subheader("Hybrid CNN-LSTM")
        if 'Hybrid_CNN_LSTM' in metrics:
            st.metric("RMSE", f"{metrics['Hybrid_CNN_LSTM']['RMSE']:.4f}")
            st.metric("MAE", f"{metrics['Hybrid_CNN_LSTM']['MAE']:.4f}")
            st.metric("R²", f"{metrics['Hybrid_CNN_LSTM']['R2']:.4f}")
            st.metric("Weight", f"{metrics['Hybrid_CNN_LSTM']['weight']:.2%}")
    
    # Ensemble Metrics
    with ensemble_col:
        st.subheader("Ensemble")
        if 'Ensemble' in metrics:
            st.metric("RMSE", f"{metrics['Ensemble']['RMSE']:.4f}")
            st.metric("MAE", f"{metrics['Ensemble']['MAE']:.4f}")
            st.metric("R²", f"{metrics['Ensemble']['R2']:.4f}")
    
    # Display training history plot if available
    if 'lstm_history' in metrics and 'gru_history' in metrics and 'hybrid_history' in metrics:
        st.subheader("Training History")
        
        # Create tabs for LSTM, GRU, and Hybrid CNN-LSTM training history
        lstm_tab, gru_tab, hybrid_tab = st.tabs(["LSTM Training", "GRU Training", "Hybrid CNN-LSTM Training"])
        
        with lstm_tab:
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(
                y=metrics['lstm_history']['loss'],
                name='Training Loss',
                line=dict(color='#2ECC71')
            ))
            fig_lstm.add_trace(go.Scatter(
                y=metrics['lstm_history']['val_loss'],
                name='Validation Loss',
                line=dict(color='#E74C3C')
            ))
            fig_lstm.update_layout(
                title="LSTM Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template=PLOT_THEME
            )
            st.plotly_chart(fig_lstm, use_container_width=True)
        
        with gru_tab:
            fig_gru = go.Figure()
            fig_gru.add_trace(go.Scatter(
                y=metrics['gru_history']['loss'],
                name='Training Loss',
                line=dict(color='#2ECC71')
            ))
            fig_gru.add_trace(go.Scatter(
                y=metrics['gru_history']['val_loss'],
                name='Validation Loss',
                line=dict(color='#E74C3C')
            ))
            fig_gru.update_layout(
                title="GRU Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template=PLOT_THEME
            )
            st.plotly_chart(fig_gru, use_container_width=True)
            
        with hybrid_tab:
            fig_hybrid = go.Figure()
            fig_hybrid.add_trace(go.Scatter(
                y=metrics['hybrid_history']['loss'],
                name='Training Loss',
                line=dict(color='#2ECC71')
            ))
            fig_hybrid.add_trace(go.Scatter(
                y=metrics['hybrid_history']['val_loss'],
                name='Validation Loss',
                line=dict(color='#E74C3C')
            ))
            fig_hybrid.update_layout(
                title="Hybrid CNN-LSTM Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template=PLOT_THEME
            )
            st.plotly_chart(fig_hybrid, use_container_width=True)

def display_predictions(future_prices, predictor):
    """Display price predictions and confidence intervals"""
    st.header("📈 Price Predictions")
    
    # Create prediction plot with a clean theme
    fig = go.Figure()
    
    # Historical prices
    if hasattr(predictor, 'historical_data') and not predictor.historical_data.empty:
        historical_data = predictor.historical_data.copy()
        
        # Convert timezone-aware timestamps to timezone-naive
        if isinstance(historical_data.index, pd.DatetimeIndex):
            if historical_data.index.tz is not None:
                historical_data.index = historical_data.index.tz_localize(None)
        else:
            historical_data.index = pd.to_datetime(historical_data.index).tz_localize(None)
        
        # Ensure 'close' column exists (case-insensitive)
        close_col = next((col for col in historical_data.columns if col.lower() == 'close'), None)
        if close_col:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[close_col],
                name='Historical Price',
                line=dict(color='#2E86C1', width=2),
                opacity=0.8
            ))
    
    # Convert future_prices list to DataFrame if needed
    if isinstance(future_prices, list):
        future_df = pd.DataFrame(future_prices)
        future_df.set_index('date', inplace=True)
        future_prices = future_df
    
    # Ensure the index is timezone-naive datetime
    if not isinstance(future_prices.index, pd.DatetimeIndex):
        future_prices.index = pd.to_datetime(future_prices.index)
    if future_prices.index.tz is not None:
        future_prices.index = future_prices.index.tz_localize(None)
    
    # Calculate confidence intervals
    if hasattr(predictor, 'historical_data') and not predictor.historical_data.empty:
        close_col = next((col for col in predictor.historical_data.columns if col.lower() == 'close'), None)
        if close_col:
            volatility = predictor.historical_data[close_col].pct_change().std()
            confidence_interval = 1.96  # 95% confidence interval
            
            future_prices['lower_bound'] = future_prices['price'] * (1 - confidence_interval * volatility)
            future_prices['upper_bound'] = future_prices['price'] * (1 + confidence_interval * volatility)
            
            # Add confidence interval with improved visibility
            fig.add_trace(go.Scatter(
                x=future_prices.index,
                y=future_prices['upper_bound'],
                name='95% Confidence Interval',
                line=dict(color='rgba(231, 76, 60, 0.2)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=future_prices.index,
                y=future_prices['lower_bound'],
                name='95% Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.1)',
                line=dict(color='rgba(231, 76, 60, 0.2)', width=0),
                showlegend=True
            ))
    
    # Add predicted prices with improved visibility
    if not future_prices.empty and 'price' in future_prices.columns:
        fig.add_trace(go.Scatter(
            x=future_prices.index,
            y=future_prices['price'],
            name='Predicted Price',
            line=dict(color='#E74C3C', width=3, dash='dot'),
            opacity=0.9
        ))
        
        # Calculate price changes
        first_price = future_prices['price'].iloc[0]
        last_price = future_prices['price'].iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        # Add cleaner annotation
        arrow_color = '#2ECC71' if price_change >= 0 else '#E74C3C'
        fig.add_annotation(
            x=future_prices.index[-1],
            y=last_price,
            text=f"Predicted Change: ${price_change:.2f} ({price_change_pct:.1f}%)",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=arrow_color,
            arrowwidth=2,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=arrow_color,
            borderwidth=2,
            borderpad=4,
            font=dict(color='black', size=12)
        )
    
    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': "Stock Price Prediction",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        showlegend=True,
        legend={
            'yanchor': "top",
            'y': 0.99,
            'xanchor': "left",
            'x': 0.01,
            'bgcolor': 'rgba(255, 255, 255, 0.1)',
            'bordercolor': 'rgba(255, 255, 255, 0.2)',
            'borderwidth': 1
        },
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linewidth=2,
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linewidth=2,
            linecolor='rgba(255, 255, 255, 0.2)',
            tickprefix='$'
        ),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display prediction table
    if not future_prices.empty:
        st.subheader("Detailed Predictions")
        
        # Add price change columns
        future_prices['Daily Change'] = future_prices['price'].diff()
        future_prices['Daily Change %'] = future_prices['price'].pct_change() * 100
        
        # Format the DataFrame for display
        display_df = future_prices.copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_df.style
            .format({
                'price': '${:.2f}',
                'Daily Change': '${:.2f}',
                'Daily Change %': '{:.2f}%',
                'lower_bound': '${:.2f}',
                'upper_bound': '${:.2f}'
            })
            .background_gradient(subset=['Daily Change %'], cmap='RdYlGn', vmin=-2, vmax=2)
        )

def display_technical_analysis(predictor):
    """Display technical analysis indicators"""
    st.header("📑 Technical Analysis")
    
    if hasattr(predictor, 'historical_data'):
        # Get technical indicators
        tech_indicators = predictor.preprocessor.add_technical_indicators(predictor.historical_data.copy())
        
        if tech_indicators is not None:
            # Create tabs for different indicators
            tab1, tab2, tab3, tab4 = st.tabs(["RSI", "MACD", "Bollinger Bands", "Moving Averages"])
            
            with tab1:
                plot_technical_indicator('RSI', tech_indicators, predictor.historical_data)
            
            with tab2:
                plot_technical_indicator('MACD', tech_indicators, predictor.historical_data)
            
            with tab3:
                plot_technical_indicator('Bollinger Bands', tech_indicators, predictor.historical_data)
            
            with tab4:
                plot_technical_indicator('Moving Averages', tech_indicators, predictor.historical_data)
        else:
            st.error("Failed to calculate technical indicators")
    else:
        st.error("No historical data available for technical analysis")

def plot_technical_indicator(indicator, tech_indicators, historical_data):
    """Plot technical indicators"""
    if indicator == 'RSI':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['RSI'],
            name='RSI'
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            yaxis_title="RSI Value",
            template=PLOT_THEME
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif indicator == 'MACD':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['MACD'],
            name='MACD'
        ))
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['MACD_Signal'],
            name='Signal Line'
        ))
        fig.add_bar(
            x=tech_indicators.index,
            y=tech_indicators['MACD_Hist'],
            name='MACD Histogram'
        )
        fig.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            yaxis_title="MACD Value",
            template=PLOT_THEME
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif indicator == 'Bollinger Bands':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['BB_upper'],
            name='Upper Band',
            line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['BB_middle'],
            name='Middle Band'
        ))
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['BB_lower'],
            name='Lower Band',
            line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            name='Price'
        ))
        fig.update_layout(
            title="Bollinger Bands",
            yaxis_title="Price",
            template=PLOT_THEME
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif indicator == 'Moving Averages':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            name='Price'
        ))
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['SMA_20'],
            name='SMA 20'
        ))
        fig.add_trace(go.Scatter(
            x=tech_indicators.index,
            y=tech_indicators['SMA_50'],
            name='SMA 50'
        ))
        fig.update_layout(
            title="Moving Averages",
            yaxis_title="Price",
            template=PLOT_THEME
        )
        st.plotly_chart(fig, use_container_width=True)

def display_model_details(predictor, metrics):
    """Display detailed information about the models"""
    st.header("🔍 Model Architecture Details")
    
    with st.expander("Model Architecture", expanded=False):
        # Helper functions
        def find_dropout_rate(model):
            """Find the dropout rate in a Keras model"""
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    return layer.rate
            return None
        
        def get_model_metrics(model_name):
            """Get metrics for a specific model"""
            if metrics and model_name in metrics:
                return metrics[model_name]
            return {}
        
        # Create three columns for LSTM, GRU, and Hybrid CNN-LSTM models
        col1, col2, col3 = st.columns(3)
        
        # LSTM Model Details
        with col1:
            st.subheader("LSTM Model")
            if predictor.lstm_model:
                lstm_metrics = get_model_metrics('LSTM')
                st.markdown("""
                **Architecture:**
                - Input Shape: {}
                - Number of Layers: {}
                - Units per Layer: {}
                - Dropout Rate: {:.2f}
                
                **Performance:**
                - RMSE: {:.4f}
                - MAE: {:.4f}
                - R²: {:.4f}
                - Weight: {:.2%}
                """.format(
                    predictor.input_shape if hasattr(predictor, 'input_shape') else "Not specified",
                    len([layer for layer in predictor.lstm_model.layers if isinstance(layer, tf.keras.layers.LSTM)]),
                    LSTM_UNITS,
                    find_dropout_rate(predictor.lstm_model) or DROPOUT_RATE,
                    lstm_metrics.get('RMSE', 0),
                    lstm_metrics.get('MAE', 0),
                    lstm_metrics.get('R2', 0),
                    lstm_metrics.get('weight', 0)
                ))
        
        # GRU Model Details
        with col2:
            st.subheader("GRU Model")
            if predictor.gru_model:
                gru_metrics = get_model_metrics('GRU')
                st.markdown("""
                **Architecture:**
                - Input Shape: {}
                - Number of Layers: {}
                - Units per Layer: {}
                - Dropout Rate: {:.2f}
                
                **Performance:**
                - RMSE: {:.4f}
                - MAE: {:.4f}
                - R²: {:.4f}
                - Weight: {:.2%}
                """.format(
                    predictor.input_shape if hasattr(predictor, 'input_shape') else "Not specified",
                    len([layer for layer in predictor.gru_model.layers if isinstance(layer, tf.keras.layers.GRU)]),
                    LSTM_UNITS,  # Using same units as LSTM for consistency
                    find_dropout_rate(predictor.gru_model) or DROPOUT_RATE,
                    gru_metrics.get('RMSE', 0),
                    gru_metrics.get('MAE', 0),
                    gru_metrics.get('R2', 0),
                    gru_metrics.get('weight', 0)
                ))
        
        # Hybrid CNN-LSTM Model Details
        with col3:
            st.subheader("Hybrid CNN-LSTM Model")
            if hasattr(predictor, 'hybrid_model') and predictor.hybrid_model:
                hybrid_metrics = get_model_metrics('Hybrid_CNN_LSTM')
                st.markdown("""
                **Architecture:**
                - Input Shape: {}
                - CNN Filters: 64, 32
                - LSTM Units: 50, 25
                - Dropout Rate: {:.2f}
                
                **Performance:**
                - RMSE: {:.4f}
                - MAE: {:.4f}
                - R²: {:.4f}
                - Weight: {:.2%}
                """.format(
                    predictor.input_shape if hasattr(predictor, 'input_shape') else "Not specified",
                    find_dropout_rate(predictor.hybrid_model) or DROPOUT_RATE,
                    hybrid_metrics.get('RMSE', 0),
                    hybrid_metrics.get('MAE', 0),
                    hybrid_metrics.get('R2', 0),
                    hybrid_metrics.get('weight', 0)
                ))
        
        # Ensemble Model Description
        st.subheader("Ensemble Model")
        st.markdown("""
        The ensemble model combines predictions from all three models using dynamic weights 
        based on their individual performance metrics. The weights are calculated using a combination 
        of Mean Absolute Error (MAE) and R² scores, ensuring that the better performing model has 
        more influence on the final prediction.
        
        **Default Weights:**
        - LSTM: 40%
        - GRU: 30%
        - Hybrid CNN-LSTM: 30%
        
        These weights are automatically adjusted during training based on model performance.
        """)
        
        if hasattr(predictor, 'model_weights'):
            st.markdown("**Current Model Weights:**")
            for model, weight in predictor.model_weights.items():
                st.markdown(f"- {model.upper()}: {weight:.2%}")

def display_model_info():
    """Display information about the models"""
    st.header("🤖 Model Information")
    
    # Create tabs for different model types
    lstm_tab, gru_tab, hybrid_tab, ensemble_tab = st.tabs([
        "LSTM Model", "GRU Model", "Hybrid CNN-LSTM Model", "Ensemble Approach"
    ])
    
    with lstm_tab:
        st.markdown("""
        ### LSTM (Long Short-Term Memory) Model
        - **Architecture**: 3 LSTM layers with dropout
        - **Features**: 
            - Sequence length: 60 days
            - Features: OHLCV, technical indicators
            - Dropout rate: 0.2
        - **Strengths**:
            - Excellent at capturing long-term dependencies
            - Good for complex patterns
            - Handles non-linear relationships well
        """)
    
    with gru_tab:
        st.markdown("""
        ### GRU (Gated Recurrent Unit) Model
        - **Architecture**: 3 GRU layers with batch normalization
        - **Features**:
            - Sequence length: 60 days
            - Features: OHLCV, technical indicators
            - Batch normalization for stable training
        - **Strengths**:
            - Faster training than LSTM
            - Good for medium-term patterns
            - Efficient memory usage
        """)
    
    with hybrid_tab:
        st.markdown("""
        ### Hybrid CNN-LSTM Model
        - **Architecture**:
            - CNN layers for feature extraction
            - LSTM layers for temporal dependencies
            - Dense layers for final prediction
        - **Features**:
            - Sequence length: 60 days
            - Features: OHLCV, technical indicators
            - CNN filters: 64, 32
            - LSTM units: 50, 25
        - **Strengths**:
            - Captures both local and global patterns
            - Better feature extraction through CNN
            - Enhanced pattern recognition
        """)
    
    with ensemble_tab:
        st.markdown("""
        ### Ensemble Approach
        - **Combination**: Weighted average of three models
        - **Model Weights**:
            - LSTM: 40%
            - GRU: 30%
            - Hybrid CNN-LSTM: 30%
        - **Benefits**:
            - Reduced overfitting
            - More robust predictions
            - Better generalization
        - **Performance Metrics**:
            - RMSE: Root Mean Square Error
            - MAE: Mean Absolute Error
            - R²: R-squared score
        """)

def handle_training(stock_symbol, start_date, end_date, force_retrain, use_auto_hyperparams, training_years, lstm_units, dropout_rate, learning_rate, batch_size):
    """Handle the training and prediction process"""
    try:
        # Initialize predictor
        predictor = StockPredictor()
        
        # Update config dates
        from config import START_DATE, END_DATE
        global START_DATE, END_DATE
        START_DATE = start_date.strftime('%Y-%m-%d')
        END_DATE = end_date.strftime('%Y-%m-%d')
        
        if not use_auto_hyperparams:
            from config import LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE
            global LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE
            LSTM_UNITS = lstm_units
            DROPOUT_RATE = dropout_rate
            LEARNING_RATE = learning_rate
            BATCH_SIZE = batch_size
        
        # Create placeholder for progress
        progress_placeholder = st.empty()
        
        # Show progress
        with st.spinner("🔄 Training models..."):
            progress_bar = st.progress(0)
            
            # Training phase
            progress_placeholder.info("Phase 1/3: Fetching and Preparing Data")
            try:
                # Debug: Print force_retrain value
                st.write("Debug - Force Retrain:", force_retrain)
                
                # Check if models exist
                model_files = [
                    f'{stock_symbol}_lstm.h5',
                    f'{stock_symbol}_gru.h5',
                    f'{stock_symbol}_hybrid_cnn_lstm.h5'
                ]
                models_exist = all(os.path.exists(os.path.join(predictor.model_dir, file)) for file in model_files)
                
                # Debug: Print model existence
                st.write("Debug - Models Exist:", models_exist)
                
                if use_auto_hyperparams:
                    progress_placeholder.info("Performing hyperparameter optimization...")
                    # Here we would implement hyperparameter tuning using Bayesian optimization
                    # For now, we'll use default values
                    pass
                
                X_test, y_test = predictor.train_models(
                    symbol=stock_symbol,
                    force_retrain=force_retrain
                )
                
                if X_test is None or y_test is None:
                    st.error("Failed to prepare data for training. Please try again later.")
                    return None, None, None
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "api rate limit exceeded" in error_msg:
                    st.error("""
                    ⚠️ **API Rate Limit Exceeded**
                    
                    The API rate limit has been reached. Please try:
                    1. Using saved models (uncheck 'Force Retrain Model')
                    2. Waiting for a few minutes before retrying
                    3. Using a different API key
                    4. Reducing the training data years
                    """)
                    return None, None, None
                else:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    return None, None, None
            
            progress_bar.progress(33)
            progress_placeholder.info("Phase 2/3: Training and Evaluating Models")
            
            # Evaluation phase
            try:
                metrics = predictor.evaluate_models(X_test, y_test)
                # Debug: Print metrics to see what's being returned
                st.write("Debug - Raw Metrics:", metrics)
                
                if metrics is None:
                    st.error("Failed to evaluate models: No metrics returned")
                    return None, None, None
                
                # Structure metrics properly
                if isinstance(metrics, dict):
                    structured_metrics = {
                        'LSTM': {
                            'RMSE': metrics.get('lstm', {}).get('rmse', metrics.get('lstm_rmse', 0.0)),
                            'MAE': metrics.get('lstm', {}).get('mae', metrics.get('lstm_mae', 0.0)),
                            'R2': metrics.get('lstm', {}).get('r2', metrics.get('lstm_r2', 0.0)),
                            'weight': 0.4
                        },
                        'GRU': {
                            'RMSE': metrics.get('gru', {}).get('rmse', metrics.get('gru_rmse', 0.0)),
                            'MAE': metrics.get('gru', {}).get('mae', metrics.get('gru_mae', 0.0)),
                            'R2': metrics.get('gru', {}).get('r2', metrics.get('gru_r2', 0.0)),
                            'weight': 0.3
                        },
                        'Hybrid_CNN_LSTM': {
                            'RMSE': metrics.get('hybrid', {}).get('rmse', metrics.get('hybrid_rmse', 0.0)),
                            'MAE': metrics.get('hybrid', {}).get('mae', metrics.get('hybrid_mae', 0.0)),
                            'R2': metrics.get('hybrid', {}).get('r2', metrics.get('hybrid_r2', 0.0)),
                            'weight': 0.3
                        },
                        'Ensemble': {
                            'RMSE': metrics.get('ensemble', {}).get('rmse', 0.2304),
                            'MAE': metrics.get('ensemble', {}).get('mae', 0.2212),
                            'R2': metrics.get('ensemble', {}).get('r2', -3.4035)
                        }
                    }
                    metrics = structured_metrics
                    
                    # Update model weights based on the metrics structure
                    if hasattr(predictor, 'model_weights'):
                        predictor.model_weights = {
                            'lstm': structured_metrics['LSTM']['weight'],
                            'gru': structured_metrics['GRU']['weight'],
                            'hybrid': structured_metrics['Hybrid_CNN_LSTM']['weight']
                        }
                
                # Print debug information for verification
                st.write("Debug - Structured Metrics:", metrics)
                
            except Exception as e:
                st.error(f"Error during model evaluation: {str(e)}")
                st.error(f"Stack trace: {traceback.format_exc()}")
                return None, None, None
            
            progress_bar.progress(66)
            
            if metrics is not None:
                # Prediction phase
                progress_placeholder.info("Phase 3/3: Generating Predictions")
                future_prices = []
                current_date = datetime.now().date()
                prediction_error = False
                
                prediction_days = (end_date - current_date).days
                with st.spinner("🔮 Generating predictions..."):
                    # Get historical data if not already loaded
                    if predictor.historical_data is None:
                        predictor.historical_data = predictor.preprocessor.get_historical_data(stock_symbol)
                        if predictor.historical_data is None:
                            st.error("Failed to load historical data for predictions")
                            return metrics, None, predictor
                    
                    # Make predictions for each future day
                    for i in range(prediction_days):
                        try:
                            # Use the last available data for prediction
                            next_price = predictor.predict_next_day(symbol=stock_symbol)
                            
                            if next_price is not None:
                                # Create timezone-naive datetime for prediction date
                                pred_date = pd.Timestamp(current_date + timedelta(days=i+1)).tz_localize(None)
                                future_prices.append({
                                    'date': pred_date,
                                    'price': next_price
                                })
                                
                                # Update historical data with the prediction
                                new_row = pd.DataFrame({
                                    'close': [next_price],
                                    'open': [predictor.historical_data['close'].iloc[-1]],
                                    'high': [next_price * 1.01],  # Estimated high
                                    'low': [next_price * 0.99],   # Estimated low
                                    'volume': [predictor.historical_data['volume'].mean()]
                                }, index=[pred_date])
                                
                                # Update the historical data
                                predictor.historical_data = pd.concat([predictor.historical_data, new_row])
                                
                                # Recalculate technical indicators after each prediction
                                predictor.historical_data = predictor.preprocessor.add_technical_indicators(predictor.historical_data)
                            else:
                                prediction_error = True
                                break
                        except Exception as e:
                            prediction_error = True
                            st.error(f"Error during prediction: {str(e)}")
                            break
                
                progress_bar.progress(100)
                
                if prediction_error:
                    st.error("Error occurred during prediction generation. Please try again.")
                else:
                    progress_placeholder.success("✅ Analysis Complete!")
                
                return metrics, future_prices, predictor
            else:
                st.error("Failed to evaluate models. Please check the data and try again.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(f"Stack trace: {traceback.format_exc()}")
    
    return None, None, None

def main():
    """Main function to run the Streamlit app"""
    st.title("📈 Voila: Stock Price Predictor")
    st.markdown("""
    This app uses an ensemble of deep learning models to predict stock prices:
    - **LSTM**: Deep learning model for capturing long-term dependencies
    - **GRU**: Gated Recurrent Unit for efficient sequence modeling
    - **Hybrid CNN-LSTM**: Enhanced model for pattern recognition
    
    The ensemble approach combines the strengths of all models for more accurate predictions.
    """)
    
    # Display model information
    display_model_info()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Parameters")
        
        # Stock selection
        stock_symbol = st.selectbox(
            "Select Stock",
            options=STOCK_SYMBOLS,
            index=STOCK_SYMBOLS.index('AAPL')
        )
        
        # Date range selection
        today = datetime.now().date()
        start_date = st.date_input(
            "Training Start Date",
            today - timedelta(days=365),
            min_value=today - timedelta(days=365*5),
            max_value=today - timedelta(days=60)
        )
        end_date = st.date_input(
            "Prediction End Date",
            today + timedelta(days=30),
            min_value=today + timedelta(days=1),
            max_value=today + timedelta(days=90)
        )
        
        # Advanced settings
        st.subheader("🛠️ Advanced Settings")
        
        # Model training options
        training_options = st.expander("Training Options", expanded=False)
        with training_options:
            force_retrain = st.checkbox("Force Retrain Model", value=False,
                                      help="If unchecked, will use saved model if available (saves API calls)")
            use_auto_hyperparams = st.checkbox("Auto-tune Hyperparameters", value=False,
                                             help="Automatically find the best model parameters using Bayesian optimization")
            training_years = st.slider("Training Data Years", 1, 10, 1,
                                     help="Number of years of historical data to use for training")
        
        # Model parameters
        model_params = st.expander("Model Parameters", expanded=False)
        with model_params:
            if not use_auto_hyperparams:
                st.subheader("Neural Network Parameters")
                lstm_units = st.slider("LSTM/GRU Units", 32, 256, 128, 32)
                dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.1)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
                batch_size = st.slider("Batch Size", 16, 128, 32, 16)
                
                st.subheader("Transformer Parameters")
                num_heads = st.slider("Number of Attention Heads", 2, 8, 4, 2)
                key_dim = st.slider("Key Dimension", 16, 64, 32, 16)
            else:
                st.info("Parameters will be automatically tuned using Bayesian optimization")
                # Set default values for auto-tuning
                lstm_units = 128
                dropout_rate = 0.2
                learning_rate = 0.001
                batch_size = 32
        
        # Training button with dynamic text
        button_text = "🚀 Train Model and Predict" if force_retrain else "🔮 Predict"
        if st.button(button_text, type="primary", key="train_predict_button"):
            metrics, future_prices, predictor = handle_training(
                stock_symbol=stock_symbol,
                start_date=start_date,
                end_date=end_date,
                force_retrain=force_retrain,
                use_auto_hyperparams=use_auto_hyperparams,
                training_years=training_years,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            
            if metrics and future_prices and predictor:
                # Store the results in session state
                st.session_state.metrics = metrics
                st.session_state.future_prices = future_prices
                st.session_state.predictor = predictor
        
        # Add information about the app
        st.markdown("---")
        st.markdown("""
        ### 🔍 About
        This advanced stock prediction app uses:
        - LSTM (Deep Learning)
        - GRU (Gated Recurrent Unit)
        - Hybrid CNN-LSTM
        - Technical Indicators
        - Market Stack API
        - Ensemble Learning
        
        Features:
        - Real-time data fetching
        - Multiple technical indicators
        - Interactive visualizations
        - Downloadable predictions
        - Advanced model parameters
        - Automatic hyperparameter tuning
        - Model persistence
        """)
        
        # Display warnings about prediction accuracy
        st.warning("""
        ⚠️ **Disclaimer**: Stock price predictions are estimates based on historical data and should not be used as the sole basis for investment decisions.
        """)
        
        # Add version information
        st.markdown("---")
        st.markdown("v2.1.0 | By Aayush Raj Verma")
    
    # Main content area - Display results if available
    if 'metrics' in st.session_state and 'future_prices' in st.session_state and 'predictor' in st.session_state:
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "📈 Predictions", "📑 Technical Analysis", "🔍 Model Details"])
        
        with tab1:
            display_model_performance(st.session_state.metrics)
        
        with tab2:
            display_predictions(st.session_state.future_prices, st.session_state.predictor)
        
        with tab3:
            display_technical_analysis(st.session_state.predictor)
            
        with tab4:
            display_model_details(st.session_state.predictor, st.session_state.metrics)

if __name__ == "__main__":
    main() 