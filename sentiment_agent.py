import logging
import datetime
import json
import os
from newsapi import NewsApiClient
from textblob import TextBlob
import time

class SentimentAgent:
    """
    An agent for fetching news and determining market sentiment with intensity.
    Includes a local caching mechanism to avoid redundant API calls.
    """
    def __init__(self, config):
        self.config = config
        self.newsapi = NewsApiClient(api_key=config['news_api']['api_key'])
        self.cache_dir = "news_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.top_constituents = [
            "Reliance Industries", "HDFC Bank", "ICICI Bank", "Infosys",
            "Larsen & Toubro", "TCS", "Bharti Airtel", "ITC", "Kotak Mahindra Bank",
            "Hindustan Unilever", "RBI", "NIFTY", "Attack", "FED", "Repo Rate", "Indian economy", "India"
        ]

    def _get_news_articles(self):
        """
        Fetches news from the last 2 days. It serves from a time-sensitive
        cache to allow for periodic refreshes of news within the same day.
        """
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=2)
        cache_file_path = os.path.join(self.cache_dir, f"news_{today.isoformat()}.json")
        CACHE_EXPIRATION_SECONDS = 3600  # 1 hour

        # Check if a recent cache file exists and is valid
        if os.path.exists(cache_file_path) and (time.time() - os.path.getmtime(cache_file_path)) < CACHE_EXPIRATION_SECONDS:
            logging.info(f"SentimentAgent: Loading recent news from cache (less than {int(CACHE_EXPIRATION_SECONDS / 60)} minutes old).")
            with open(cache_file_path, 'r') as f:
                return json.load(f)

        logging.info("SentimentAgent: Fetching fresh news from API for the last 2 days...")
        try:
            query = f"({self.config['trading_flags']['underlying_instrument']}) OR " + " OR ".join(f'"{c}"' for c in self.top_constituents)
            top_headlines = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',  # Sort by newest first to prioritize recent news
                page_size=100,
                from_param=from_date.isoformat(),
                to=today.isoformat()
            )
            with open(cache_file_path, 'w') as f:
                json.dump(top_headlines, f)
            return top_headlines
        except Exception as e:
            logging.error(f"SentimentAgent: Could not fetch news from API: {e}")
            return None

    def get_market_sentiment(self):
        """
        Calculates sentiment using a weighted average based on news recency
        and returns bias with intensity.
        """
        top_headlines = self._get_news_articles() #
        if not top_headlines or not top_headlines.get('articles'):
            logging.warning("SentimentAgent: No news articles found.")
            return "Neutral" #

        sentiment_scores = []
        for article in top_headlines['articles']:
            if (title := article.get('title', '')) and title != "[Removed]":
                content = f"{title}. {article.get('description', '')}"
                sentiment_scores.append(TextBlob(content).sentiment.polarity) #

        if not sentiment_scores:
            logging.warning("SentimentAgent: No valid headlines to analyze.")
            return "Neutral" #

        # --- Weighted Average Calculation ---
        # Articles are sorted newest to oldest from the API call.
        weighted_sum = 0
        total_weight = 0
        n = len(sentiment_scores)

        for i, score in enumerate(sentiment_scores):
            # Weight decays linearly from n for the newest to 1 for the oldest.
            weight = n - i
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            avg_sentiment = 0.0
        else:
            avg_sentiment = weighted_sum / total_weight

        logging.info(f"SentimentAgent: Weighted average sentiment score is {avg_sentiment:.3f}")

        if avg_sentiment > 0.4:
            return "Very Bullish" #
        elif avg_sentiment > 0.05:
            return "Bullish" #
        elif avg_sentiment < -0.4:
            return "Very Bearish" #
        elif avg_sentiment < -0.05:
            return "Bearish" #
        else:
            return "Neutral" #