import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import *

class StockVisualizer:
    def __init__(self):
        self.theme = PLOT_THEME
        self.height = PLOT_HEIGHT
        self.width = PLOT_WIDTH
    
    def plot_stock_data(self, df, title="Stock Price Data"):
        """Plot historical stock data with candlesticks"""
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title="Price",
            xaxis_title="Date",
            template=self.theme,
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def plot_predictions(self, actual, predicted, dates, title="Stock Price Predictions"):
        """Plot actual vs predicted stock prices"""
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            name="Actual",
            line=dict(color='blue')
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted,
            name="Predicted",
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Price",
            xaxis_title="Date",
            template=self.theme,
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def plot_technical_indicators(self, df):
        """Plot technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Price", "RSI", "MACD")
        )
        
        # Price and Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_high'],
                name="BB Upper",
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_low'],
                name="BB Lower",
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name="RSI",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                name="MACD",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Technical Indicators",
            template=self.theme,
            height=self.height * 1.5,
            width=self.width,
            showlegend=True
        )
        
        return fig
    
    def plot_model_metrics(self, metrics_history):
        """Plot model training metrics"""
        fig = go.Figure()
        
        for metric, values in metrics_history.items():
            fig.add_trace(go.Scatter(
                y=values,
                name=metric,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Model Training Metrics",
            yaxis_title="Value",
            xaxis_title="Epoch",
            template=self.theme,
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance):
        """Plot feature importance for XGBoost model"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values())
        ))
        
        fig.update_layout(
            title="Feature Importance",
            yaxis_title="Importance Score",
            xaxis_title="Feature",
            template=self.theme,
            height=self.height,
            width=self.width,
            xaxis_tickangle=45
        )
        
        return fig 