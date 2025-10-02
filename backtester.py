import pandas as pd
import datetime
import logging
from strategy_factory import get_strategy
from market_context import MarketConditionIdentifier
import time
import pandas_ta as ta
import os # Import os

def fetch_historical_data_in_chunks(kite, token, from_date, to_date, timeframe):
    """
    Fetches historical data by breaking the request into smaller chunks
    to comply with the API's date range limits for intraday data.
    """
    all_data = []
    current_from = from_date
    
    while current_from <= to_date:
        current_to = current_from + datetime.timedelta(days=99)
        if current_to > to_date:
            current_to = to_date
            
        logging.info(f"Fetching data from {current_from} to {current_to}")
        try:
            chunk = kite.historical_data(token, current_from, current_to, timeframe)
            all_data.extend(chunk)
            time.sleep(0.5) 
        except Exception as e:
            logging.error(f"Error fetching data chunk from {current_from} to {current_to}: {e}")
        
        current_from = current_to + datetime.timedelta(days=1)
        
    return pd.DataFrame(all_data)

def run_backtest(kite, config, strategy_name, from_date, to_date, target_conditions=None):
    """
    Runs a backtest for a given strategy. If target_conditions are provided,
    it runs a conditional backtest only on days matching those conditions.
    """
    mode = "Conditional" if target_conditions else "Full"
    logging.info(f"--- Starting {mode} Backtest for Strategy: {strategy_name} ---")
    if mode == "Conditional": logging.info(f"--- Target Conditions: {target_conditions} ---")

    underlying_name = config['trading_flags']['underlying_instrument']
    timeframe = config['trading_flags']['chart_timeframe']
    
    try:
        nifty_token = [i['instrument_token'] for i in kite.instruments('NSE') if i['tradingsymbol'] == underlying_name][0]
        vix_token = [i['instrument_token'] for i in kite.instruments('NSE') if i['tradingsymbol'] == 'INDIA VIX'][0]
    except (IndexError, KeyError) as e:
        logging.error(f"Could not find instrument token: {e}"); return 0.0

    all_data_day = fetch_historical_data_in_chunks(kite, nifty_token, from_date, to_date, "day")
    all_data_tf = fetch_historical_data_in_chunks(kite, nifty_token, from_date, to_date, timeframe)
    vix_data = fetch_historical_data_in_chunks(kite, vix_token, from_date, to_date, "day")
    
    if all_data_day.empty or all_data_tf.empty:
        logging.error("Failed to fetch sufficient historical data for backtest."); return 0.0

    all_data_day['date'] = pd.to_datetime(all_data_day['date']).dt.date
    vix_data['date'] = pd.to_datetime(vix_data['date']).dt.date
    all_data_day['daily_volatility'] = all_data_day['close'].pct_change().rolling(window=7).std()
    all_data_tf['date_only'] = pd.to_datetime(all_data_tf['date']).dt.date

    strategy = get_strategy(strategy_name, kite, config)
    trades = []
    
    historical_dates_to_test = []
    if target_conditions:
        condition_identifier = MarketConditionIdentifier()
        logging.info("Filtering historical dates based on target conditions...")
        for date_obj in all_data_day['date'].unique():
            hist_conditions = condition_identifier.get_conditions_for_date(date_obj, vix_data, all_data_day)
            if target_conditions.issubset(hist_conditions):
                historical_dates_to_test.append(date_obj)
        logging.info(f"Found {len(historical_dates_to_test)} matching historical days for conditional backtest.")
    else:
        historical_dates_to_test = all_data_day['date'].unique().tolist()

    if not historical_dates_to_test:
        logging.warning("No matching historical days found for conditional backtest."); return 0.0

    for i in range(1, len(all_data_day)):
        current_date = all_data_day.iloc[i]['date']
        if current_date not in historical_dates_to_test: continue

        day_tf_df = all_data_tf[all_data_tf['date_only'] == current_date].copy()
        if len(day_tf_df) < 50: continue

        # Set DatetimeIndex to prevent VWAP error
        day_tf_df['date'] = pd.to_datetime(day_tf_df['date'])
        day_tf_df = day_tf_df.set_index('date').sort_index()

        from indicators import calculate_cpr
        cpr_pivots = calculate_cpr(all_data_day.iloc[i-1:i])
        if not cpr_pivots: continue
        cpr_pivots['prev_high'] = all_data_day.iloc[i-1]['high'].item()
        cpr_pivots['prev_low'] = all_data_day.iloc[i-1]['low'].item()
        
        # Pre-calculate all indicators for the day
        day_tf_df.ta.vwap(append=True)
        day_tf_df['rsi'] = ta.rsi(day_tf_df['close'])
        day_tf_df['ema_20'] = ta.ema(day_tf_df['close'], length=20)
        day_tf_df['ema_50'] = ta.ema(day_tf_df['close'], length=50)
        supertrend = ta.supertrend(day_tf_df['high'], day_tf_df['low'], day_tf_df['close'], length=10, multiplier=3)
        if supertrend is not None and not supertrend.empty: day_tf_df['supertrend_direction'] = supertrend.get('SUPERTd_10_3.0', 0)
        macd = ta.macd(day_tf_df['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            day_tf_df['macd'] = macd.get('MACD_12_26_9', 0)
            day_tf_df['macd_signal'] = macd.get('MACDs_12_26_9', 0)
        day_tf_df['atr'] = ta.atr(day_tf_df['high'], day_tf_df['low'], day_tf_df['close'], length=14)
        day_tf_df['atr_ma'] = day_tf_df['atr'].rolling(window=20).mean()
        day_tf_df['spread'] = day_tf_df['high'] - day_tf_df['low']
        day_tf_df['volume_ma'] = day_tf_df['volume'].rolling(window=20).mean()
        
        sentiment = "Bullish"
        position = None
        for j in range(20, len(day_tf_df)):
            if not position:
                signal = strategy.generate_signals(day_tf_df, j, sentiment, cpr_pivots=cpr_pivots)
                if signal != 'HOLD':
                    position, entry_price = signal, day_tf_df.iloc[j]['close']
            else:
                if (position == 'BUY' and day_tf_df.iloc[j]['low'] < entry_price * 0.98) or \
                   (position == 'SELL' and day_tf_df.iloc[j]['high'] > entry_price * 1.02):
                    trades.append({'entry': entry_price, 'exit': day_tf_df.iloc[j]['close'], 'type': position})
                    position = None
    
    if not trades:
        logging.warning(f"No trades were executed during {mode} backtest for {strategy_name}."); return 0.0

    wins = sum(1 for t in trades if (t['type'] == 'BUY' and t['exit'] > t['entry']) or (t['type'] == 'SELL' and t['exit'] < t['entry']))
    win_rate = (wins / len(trades)) * 100
    logging.info(f"--- {mode} Backtest Results for {strategy_name}: ---")
    logging.info(f"Total Trades: {len(trades)}, Wins: {wins}, Win Rate: {win_rate:.2f}%")

    # --- NEW: Save backtest results to CSV ---
    result_data = {
        'timestamp': [datetime.datetime.now()],
        'strategy': [strategy_name],
        'mode': [mode],
        'conditions': [str(target_conditions) if target_conditions else 'N/A'],
        'from_date': [from_date],
        'to_date': [to_date],
        'total_trades': [len(trades)],
        'win_rate_pct': [win_rate]
    }
    result_df = pd.DataFrame(result_data)
    log_path = 'output/backtest_results.csv'
    if not os.path.exists(log_path):
        result_df.to_csv(log_path, index=False)
    else:
        result_df.to_csv(log_path, mode='a', header=False, index=False)
    logging.info(f"Backtest results for {strategy_name} saved to {log_path}")
    # --- END NEW ---
    
    return win_rate