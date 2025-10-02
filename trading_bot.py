import logging
import yaml
import time
import datetime
import calendar
import pandas as pd
import asyncio
from kiteconnect import KiteConnect, exceptions
from agents import OrderExecutionAgent, PositionManagementAgent
from sentiment_agent import SentimentAgent
from langgraph_agent import LangGraphAgent
from strategy_factory import get_strategy
from backtester import run_backtest
from reporting import send_daily_report, initialize_trade_log, log_trade, send_monthly_report
from indicators import calculate_cpr, is_trend_overextended, check_momentum_divergence
from indicator_calculator import calculate_all_indicators
from market_context import MarketConditionIdentifier
from rag_service import RAGService
import multiprocessing
import warnings

warnings.filterwarnings(
    "once",
    category=UserWarning,
    message=".*Converting to PeriodArray/Index representation will drop timezone information.*"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open('config.yaml', 'r') as file: return yaml.safe_load(file)

def save_config(config):
    with open('config.yaml', 'w') as file: yaml.dump(config, file)

class TradingBotOrchestrator:
    def __init__(self, config):
        self.config = config
        self.kite = KiteConnect(api_key=config['zerodha']['api_key'], timeout=120, debug=True)
        self.active_strategy_name = "None"
        self.active_strategy = None
        self.rag_service = RAGService(config)
        self.langgraph_agent = LangGraphAgent(config, self.rag_service)
        self.sentiment_agent = SentimentAgent(config)
        
        self.market_condition_identifier = None
        self.order_agent = None
        self.position_agent = None
        
        self.day_sentiment = ""
        self.trades_today_count = 0
        self.no_trade_reason = None
        self.bot_state = "STARTING"
        self.last_processed_timestamp = None
        self.last_divergence_warning = ""
        self.awaiting_signal_since = None

    def authenticate(self):
        logging.info("Attempting fresh authentication...")
        logging.info(f"Login URL: {self.kite.login_url()}")
        request_token = input("Enter request_token: ")
        try:
            data = self.kite.generate_session(request_token, api_secret=self.config['zerodha']['api_secret'])
            access_token = data['access_token']
            
            self.kite.set_access_token(access_token)
            self.config['zerodha']['access_token'] = access_token
            save_config(self.config)
            
            profile = self.kite.profile()
            logging.info(f"Authentication successful. Connected as {profile.get('user_name', 'user')}.")
            
            logging.info("Initializing session-dependent agents...")
            self.market_condition_identifier = MarketConditionIdentifier(self.kite, self.config)
            self.order_agent = OrderExecutionAgent(self.kite, self.config)
            self.position_agent = PositionManagementAgent(self.kite, self.config, self.rag_service)
            logging.info("Agents initialized successfully.")
            
            return True
        except Exception as e:
            logging.error(f"Auth failed: {e}", exc_info=True); return False
            
    # Helper function to check if the market is open ---
    def is_market_open(self):
        """Checks if the current time is within Indian market hours."""
        now = datetime.datetime.now().time()
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        # Also check if it's a weekday
        return market_open <= now <= market_close and datetime.datetime.now().weekday() < 5

    # Helper function to get the next trading day ---
    def get_next_trading_day(self):
        """Calculates the next weekday."""
        today = datetime.date.today()
        next_day = today + datetime.timedelta(days=1)
        while next_day.weekday() >= 5: # 5 = Saturday, 6 = Sunday
            next_day += datetime.timedelta(days=1)
        return next_day

    async def setup(self):
        self.bot_state = "SETUP"
        logging.info("--- Starting Bot Setup Sequence ---")
        today = datetime.date.today()
        
        try:
            todays_conditions = self.market_condition_identifier.get_conditions_for_date(today)
            if 'UNKNOWN' in todays_conditions:
                self.no_trade_reason = "Could not determine market conditions."; return False

            if self.config['trading_flags'].get('manual_sentiment_override', False):
                self.day_sentiment = self._get_manual_sentiment_input("Manual sentiment override is ACTIVE.")
            else:
                self.day_sentiment = self.sentiment_agent.get_market_sentiment()
                if self.day_sentiment == "Neutral":
                    self.day_sentiment = self._get_manual_sentiment_input("Automated sentiment is 'Neutral'. Manual override required.")

            if self.day_sentiment == "Neutral":
                self.no_trade_reason = "Market sentiment is Neutral (confirmed manually)."; return False
            
            logging.info(f"Today's Market Conditions: {todays_conditions} | Final Sentiment: {self.day_sentiment}")

            user_prompt = ""
            if self.config['trading_flags'].get('enable_natural_language_prompt', False):
                user_prompt = input("Enter your trading observation or preference (or press Enter): ")

            rag_context = None
            use_rag_flag = self.config['trading_flags'].get('use_rag', False)
            rag_min_days = self.config['trading_flags'].get('rag_min_trading_days', 5)
            
            if use_rag_flag:
                trade_log_df = self.rag_service._load_data(self.rag_service.trade_log_path)
                if trade_log_df is not None and not trade_log_df.empty:
                    trading_days = pd.to_datetime(trade_log_df['Timestamp']).dt.date.nunique()
                    if trading_days >= rag_min_days:
                        logging.info(f"Sufficient historical data found ({trading_days} days). Activating RAG.")
                        rag_context = self.rag_service.retrieve_context_for_strategy_selection(todays_conditions)
                    else:
                        logging.warning(f"RAG disabled: Insufficient historical data. Found {trading_days} days, need {rag_min_days}.")
                else:
                    logging.warning("RAG disabled: No trade log found.")
            else:
                logging.info("RAG is disabled in config.yaml.")

            best_strategy_name = await self.langgraph_agent.get_recommended_strategy(todays_conditions, user_prompt, rag_context)
            
            self.active_strategy_name = best_strategy_name
            self.active_strategy = get_strategy(best_strategy_name, self.kite, self.config)
            
            initialize_trade_log()
            
            token = self.order_agent.underlying_token
            hist = await asyncio.to_thread(self.kite.historical_data, token, today - datetime.timedelta(days=7), today, "day")
            prev_day_data = pd.DataFrame(hist).iloc[-2:-1]
            self.position_agent.cpr_pivots = calculate_cpr(prev_day_data)
            
            self.bot_state = "AWAITING_SIGNAL"
            self.awaiting_signal_since = datetime.datetime.now()
            logging.info(f"Setup complete. Active strategy: '{self.active_strategy.name}'. Waiting for signal...")
            return True
        except Exception as e:
            logging.error(f"Setup failed: {e}", exc_info=True)
            self.no_trade_reason = str(e)
            return False

    def _get_manual_sentiment_input(self, reason: str):
        logging.warning(reason)
        valid_sentiments = ["Very Bullish", "Bullish", "Bearish", "Very Bearish", "Neutral"]
        while True:
            prompt = f"Please enter market sentiment {valid_sentiments}: "
            manual_input = input(prompt)
            if manual_input in valid_sentiments: return manual_input
            logging.warning(f"Invalid input. Please choose from {valid_sentiments}.")

    async def run(self):
        # --- NEW: Check if market is closed at the very start ---
        if not self.is_market_open():
            logging.warning("Market is currently closed.")
            try:
                # Fetch last day's NIFTY 50 data
                token = self.order_agent.underlying_token
                to_date = datetime.date.today()
                from_date = to_date - datetime.timedelta(days=7) # Fetch a week to ensure we get the last trading day
                hist_data = await asyncio.to_thread(self.kite.historical_data, token, from_date, to_date, "day")
                
                if hist_data:
                    last_day = hist_data[-1]
                    print("\n--- Last Trading Day Summary ---")
                    print(f"Date:   {last_day['date'].strftime('%A, %d %B %Y')}")
                    print(f"Open:   {last_day['open']:.2f}")
                    print(f"High:   {last_day['high']:.2f}")
                    print(f"Low:    {last_day['low']:.2f}")
                    print(f"Close:  {last_day['close']:.2f}")
                    print("---------------------------------")

                # Fetch latest news
                news = self.sentiment_agent._get_news_articles()
                if news and news.get('articles'):
                    print("\n--- Latest News Headlines ---")
                    for article in news['articles'][:5]: # Show top 5
                        print(f"- {article['title']}")
                    print("---------------------------------")

            except Exception as e:
                logging.error(f"Could not fetch post-market data: {e}")
            
            next_day_str = self.get_next_trading_day().strftime('%A, %d %B')
            print(f"\nMarket is closed right now, enjoy your day and come back on {next_day_str} at 9:15 AM to trade like a Warrior!\n")
            return # Exit the run method
        # --- END NEW ---

        try:
            margins = await asyncio.to_thread(self.kite.margins)
            self.starting_capital = margins['equity']['available']['live_balance']
        except Exception as e:
            logging.error(f"Could not fetch starting capital: {e}")
            self.starting_capital = 0

        if not await self.setup():
            logging.warning(f"Setup failed. Reason: {self.no_trade_reason or 'Unknown'}. Bot will exit.")
            send_daily_report(self.config, str(datetime.date.today()), no_trades_reason=self.no_trade_reason)
            return

        is_paper = self.config['trading_flags']['paper_trading']
        logging.info(f"Bot running in {'PAPER TRADING' if is_paper else 'LIVE TRADING'} mode.")

        while self.is_market_open(): # Changed loop condition to re-check market status
            try:
                if self.bot_state == "AWAITING_SIGNAL":
                    reassessment_period = self.config['trading_flags'].get('strategy_reassessment_period_minutes', 60)
                    if self.awaiting_signal_since and (datetime.datetime.now() - self.awaiting_signal_since).total_seconds() > reassessment_period * 60:
                        logging.warning(f"No trade signal for over {reassessment_period} minutes. Re-assessing strategy...")
                        if await self.setup():
                            continue
                        else:
                            self.bot_state = "STOPPED"
                            continue
                    
                    if self.trades_today_count >= self.config['trading_flags']['max_trades_per_day']:
                        self.bot_state = "STOPPED"; continue
                    
                    token = self.order_agent.underlying_token
                    hist_df = pd.DataFrame(await asyncio.to_thread(self.kite.historical_data, token, datetime.datetime.now() - datetime.timedelta(days=5), datetime.datetime.now(), self.config['trading_flags']['chart_timeframe']))
                    hist_df['date'] = pd.to_datetime(hist_df['date'])
                    hist_df.set_index('date', inplace=True)
                    
                    day_df_for_status = calculate_all_indicators(hist_df.copy(), self.config)
                    status_message = self.active_strategy.get_status_message(day_df_for_status, self.day_sentiment, cpr_pivots=self.position_agent.cpr_pivots)
                    logging.info(status_message)

                    latest_candle_timestamp = hist_df.index[-1]
                    if self.last_processed_timestamp is None or latest_candle_timestamp > self.last_processed_timestamp:
                        self.last_processed_timestamp = latest_candle_timestamp
                    
                        signal = self.active_strategy.generate_signals(day_df_for_status, self.day_sentiment, cpr_pivots=self.position_agent.cpr_pivots)
                        
                        if signal != 'HOLD':
                            is_primary_signal = (signal == 'BUY' and self.day_sentiment in ['Bullish', 'Very Bullish']) or \
                                                (signal == 'SELL' and self.day_sentiment in ['Bearish', 'Very Bearish'])
                            
                            if getattr(self.active_strategy, 'is_reversal_trade', False) or is_primary_signal:
                                logging.info(f"PRIMARY SIGNAL '{signal}' detected, proceeding with trade.")
                                trade_details = await (self.order_agent.get_paper_trade_details(signal) if is_paper else self.order_agent.place_trade(signal))
                                if trade_details:
                                    trade_details['Strategy'] = self.active_strategy_name
                                    self.position_agent.start_trade(trade_details)
                                    self.trades_today_count += 1
                                    self.bot_state = "IN_POSITION"
                                    self.awaiting_signal_since = None 
                            else:
                                logging.warning(f"COUNTER-SIGNAL DETECTED: A '{signal}' signal occurred when sentiment is '{self.day_sentiment}'.")

                elif self.bot_state == "IN_POSITION":
                    token = self.order_agent.underlying_token
                    underlying_df_hist = pd.DataFrame(await asyncio.to_thread(self.kite.historical_data, token, datetime.datetime.now() - datetime.timedelta(days=1), datetime.datetime.now(), self.config['trading_flags']['chart_timeframe']))
                    underlying_df_hist['date'] = pd.to_datetime(underlying_df_hist['date'])
                    underlying_df_hist.set_index('date', inplace=True)
                    underlying_df = calculate_all_indicators(underlying_df_hist, self.config)
                    status = await self.position_agent.manage(is_paper, underlying_hist_df=underlying_df, sentiment_agent=self.sentiment_agent, gemini_api_key=self.config.get('google_api', {}).get('api_key'))
                    if status and status != 'ACTIVE':
                        log_trade(status)
                        self.bot_state = "AWAITING_SIGNAL"
                        self.awaiting_signal_since = datetime.datetime.now()

                await asyncio.sleep(30)
            except Exception as e:
                logging.error(f"Error in main loop: {e}", exc_info=True); await asyncio.sleep(60)
        
        logging.info("Market is now closed. Shutting down trading loop.")
        today = datetime.date.today()
        send_daily_report(self.config, str(today), starting_capital=self.starting_capital)
        if today.day == calendar.monthrange(today.year, today.month)[1]:
            send_monthly_report(self.config, str(today))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    bot = TradingBotOrchestrator(load_config())
    if bot.authenticate():
        asyncio.run(bot.run())
