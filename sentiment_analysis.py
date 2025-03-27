import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
from config import ALPHA_VANTAGE_API_KEY

class SentimentAnalyzer:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
    
    def get_news_sentiment(self, symbol):
        """Fetch news and analyze sentiment for a given stock symbol."""
        try:
            # Fetch news from Alpha Vantage News API
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}'
            response = requests.get(url)
            data = response.json()
            
            if 'feed' not in data:
                return None
            
            sentiments = []
            for article in data['feed']:
                # Get article sentiment if available from API
                if 'overall_sentiment_score' in article:
                    sentiment_score = float(article['overall_sentiment_score'])
                else:
                    # Calculate sentiment using TextBlob if not provided
                    text = article['title'] + " " + article.get('summary', '')
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity
                
                sentiments.append({
                    'date': article['time_published'][:10],
                    'sentiment': sentiment_score
                })
            
            # Convert to DataFrame and calculate daily average sentiment
            if sentiments:
                df = pd.DataFrame(sentiments)
                df['date'] = pd.to_datetime(df['date'])
                daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
                return daily_sentiment
            
            return None
            
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return None
    
    def get_social_media_sentiment(self, symbol):
        """
        Placeholder for social media sentiment analysis.
        This could be implemented using Twitter/Reddit APIs in the future.
        """
        pass
    
    def calculate_market_sentiment(self, symbol):
        """Calculate overall market sentiment score."""
        news_sentiment = self.get_news_sentiment(symbol)
        
        if news_sentiment is not None:
            # For now, just use news sentiment
            # Could be extended to include social media sentiment
            return news_sentiment
        
        return None 