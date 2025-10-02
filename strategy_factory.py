import logging
import pandas as pd
import pandas_ta as ta
import datetime
from indicators import (
    calculate_cpr, calculate_rsi, check_rsi_divergence, 
    check_cpr_breakout, calculate_ema, check_ema_crossover,
    check_momentum_divergence, is_trend_overextended
)

class BaseStrategy:
    """Base class for all trading strategies."""
    def __init__(self, kite, config):
        self.kite = kite
        self.config = config
        self.name = "Base"
        self.is_reversal_trade = False # Default behavior requires sentiment confirmation


    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        """
        Generates 'BUY', 'SELL', or 'HOLD' signals.
        The 'index' argument is now optional. If None, it defaults to the latest candle.
        """
        raise NotImplementedError

    def get_status_message(self, day_df, sentiment, **kwargs):
        """Returns a human-readable status message."""
        return "Awaiting signal: Generic strategy waiting for conditions."

class Gemini_Default_Strategy(BaseStrategy):
    """The original Gemini strategy based on CPR, EMA, and RSI."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Gemini_Default"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'
        if 'ema_50' not in day_df.columns:
            day_df['ema_50'] = calculate_ema(day_df['close'], 50)
        if 'rsi' not in day_df.columns:
            day_df['rsi'] = calculate_rsi(day_df['close'], 14)

        cpr_pivots = kwargs.get('cpr_pivots', {})
        current_candle = day_df.iloc[index]
        
        primary_signal_met = False
        confirmation_signals_met = 0

        cpr_breakout_signal = check_cpr_breakout(current_candle, cpr_pivots, day_df.iloc[index-1])
        if cpr_breakout_signal == sentiment:
            primary_signal_met = True
        
        if primary_signal_met:
            if sentiment == 'Bullish':
                if current_candle['close'] > current_candle['ema_50']: confirmation_signals_met += 1
                if current_candle['rsi'] > 55: confirmation_signals_met += 1
            elif sentiment == 'Bearish':
                if current_candle['close'] < current_candle['ema_50']: confirmation_signals_met += 1
                if current_candle['rsi'] < 45: confirmation_signals_met += 1
        
        logging.debug(f"[{self.name}] Check on {current_candle.name}: Primary Met={primary_signal_met}, Confirmations Met={confirmation_signals_met}")

        if primary_signal_met and confirmation_signals_met >= 1:
            logging.info(f"[{self.name}] Signal confirmed: Primary condition and {confirmation_signals_met} confirmation(s) met.")
            return 'BUY' if sentiment == 'Bullish' else 'SELL'
        
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        cpr = kwargs.get('cpr_pivots', {})
        if not cpr or 'tc' not in cpr or 'bc' not in cpr:
            return f"Awaiting signal for {self.name}: CPR pivots not yet calculated."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price to cross above CPR Top ({cpr['tc']:.2f}) and be confirmed by EMA(50) & RSI > 55."
        else:
            return f"Awaiting SELL signal: Price to cross below CPR Bottom ({cpr['bc']:.2f}) and be confirmed by EMA(50) & RSI < 45."

class Supertrend_MACD_Strategy(BaseStrategy):
    """A trend-following strategy based on Supertrend and MACD."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Supertrend_MACD"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'
        
        if 'supertrend_direction' not in day_df.columns:
            supertrend = ta.supertrend(day_df['high'], day_df['low'], day_df['close'])
            if supertrend is not None and not supertrend.empty:
                day_df['supertrend_direction'] = supertrend.get('SUPERTd_7_3.0')
        if 'macd' not in day_df.columns:
            macd = ta.macd(day_df['close'])
            if macd is not None and not macd.empty:
                day_df[['macd', 'macd_signal']] = macd[['MACD_12_26_9', 'MACDs_12_26_9']]

        current = day_df.iloc[index]
         # --- MODIFIED LOGIC: Check for both BUY and SELL signals ---
        is_bullish_signal = current.get('supertrend_direction') == 1 and current.get('macd') > current.get('macd_signal')
        is_bearish_signal = current.get('supertrend_direction') == -1 and current.get('macd') < current.get('macd_signal')

        if is_bullish_signal:
            logging.info(f"[{self.name}] BUY Signal condition met.")
            return 'BUY'
        if is_bearish_signal:
            logging.info(f"[{self.name}] SELL Signal condition met.")
            return 'SELL'
            
        return 'HOLD'
    
    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Supertrend must be bullish AND the MACD line must cross above its signal line."
        else:
            return f"Awaiting SELL signal: Supertrend must be bearish AND the MACD line must cross below its signal line."

class VolatilityClusterStrategy(BaseStrategy):
    """A reversal strategy based on the concept of Volatility Clustering."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Volatility_Cluster_Reversal"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 20: return 'HOLD'

        if 'atr' not in day_df.columns:
            day_df['atr'] = ta.atr(day_df['high'], day_df['low'], day_df['close'], length=14)
        if 'atr_ma' not in day_df.columns:
            day_df['atr_ma'] = day_df['atr'].rolling(window=20).mean()
            
        last_completed_candle = day_df.iloc[index - 1]

        if pd.isna(last_completed_candle['atr']) or pd.isna(last_completed_candle['atr_ma']): return 'HOLD'

        is_high_volatility = last_completed_candle['atr'] > last_completed_candle['atr_ma']
        avg_candle_size = day_df['atr'].iloc[index-1]
        last_candle_size = abs(last_completed_candle['open'] - last_completed_candle['close'])
        is_large_move = last_candle_size > (avg_candle_size * 1.5)

        if sentiment in ['Bullish', 'Very Bullish']:
            is_reversal_candle = last_completed_candle['close'] < last_completed_candle['open']
            if is_high_volatility and is_large_move and is_reversal_candle:
                logging.info(f"[{self.name}] Reversal BUY signal: High volatility detected after a large down move.")
                return 'BUY'
        elif sentiment in ['Bearish', 'Very Bearish']:
            is_reversal_candle = last_completed_candle['close'] > last_completed_candle['open']
            if is_high_volatility and is_large_move and is_reversal_candle:
                logging.info(f"[{self.name}] Reversal SELL signal: High volatility detected after a large up move.")
                return 'SELL'
            
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Looking for a large downward candle during high volatility to signal a potential reversal up."
        else:
            return f"Awaiting SELL signal: Looking for a large upward candle during high volatility to signal a potential reversal down."

class VSA_Strategy(BaseStrategy):
    """A strategy based on Volume Spread Analysis (VSA)."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Volume_Spread_Analysis"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 20: return 'HOLD'
        
        if 'volume_ma' not in day_df.columns:
            day_df['volume_ma'] = day_df['volume'].rolling(window=20).mean()
        if 'spread' not in day_df.columns:
            day_df['spread'] = day_df['high'] - day_df['low']
        
        last_candle = day_df.iloc[index - 1]
        
        is_high_volume = last_candle.get('volume', 0) > (last_candle.get('volume_ma', 0) * 1.3)
        is_wide_spread = last_candle.get('spread', 0) > day_df['spread'].rolling(window=20).mean().iloc[index - 1]
        
        if sentiment in ['Bullish', 'Very Bullish']:
            is_down_bar = last_candle['close'] < last_candle['open']
            is_high_close = last_candle['close'] > (last_candle['low'] + last_candle['spread'] * 0.5)
            if is_down_bar and is_high_volume and is_wide_spread and is_high_close:
                logging.info(f"[{self.name}] Signal confirmed: Sign of Strength detected."); return 'BUY'
        
        if sentiment in ['Bearish', 'Very Bearish']:
            is_up_bar = last_candle['close'] > last_candle['open']
            is_low_close = last_candle['close'] < (last_candle['low'] + last_candle['spread'] * 0.5)
            if is_up_bar and is_high_volume and is_wide_spread and is_low_close:
                logging.info(f"[{self.name}] Signal confirmed: Sign of Weakness detected."); return 'SELL'
            
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Looking for a 'Sign of Strength' - a down-bar with high volume closing in its upper half."
        else:
            return f"Awaiting SELL signal: Looking for a 'Sign of Weakness' - an up-bar with high volume closing in its lower half."

class Momentum_VWAP_RSI_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "Momentum_VWAP_RSI"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'
        
        current = day_df.iloc[index]
        if sentiment in ['Bullish', 'Very Bullish'] and current['close'] > current['vwap'] and current['rsi'] > 55: return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and current['close'] < current['vwap'] and current['rsi'] < 45: return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        vwap = day_df.iloc[-1].get('vwap', 0)
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price needs to be above VWAP ({vwap:.2f}) with RSI > 55."
        else:
            return f"Awaiting SELL signal: Price needs to be below VWAP ({vwap:.2f}) with RSI < 45."

class Breakout_Prev_Day_HL_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "Breakout_Prev_Day_HL"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'

        cpr = kwargs.get('cpr_pivots', {})
        pdh, pdl = cpr.get('prev_high'), cpr.get('prev_low')
        if not pdh or not pdl: return 'HOLD'
        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        if sentiment in ['Bullish', 'Very Bullish'] and last['close'] < pdh and current['close'] > pdh and current['volume'] > (current['volume_ma'] * 1.2): return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and last['close'] > pdl and current['close'] < pdl and current['volume'] > (current['volume_ma'] * 1.2): return 'SELL'
        return 'HOLD'
    
    def get_status_message(self, day_df, sentiment, **kwargs):
        cpr = kwargs.get('cpr_pivots', {})
        pdh, pdl = cpr.get('prev_high'), cpr.get('prev_low')
        if not pdh or not pdl:
            return f"Awaiting signal for {self.name}: Previous day's high/low not available."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price needs to break above previous day's high ({pdh:.2f}) on high volume."
        else:
            return f"Awaiting SELL signal: Price needs to break below previous day's low ({pdl:.2f}) on high volume."

class Opening_Range_Breakout_Strategy(BaseStrategy):
    def __init__(self, kite, config):
        super().__init__(kite, config); self.name = "Opening_Range_Breakout"
        self.orb_high = None; self.orb_low = None; self.orb_period_set = False
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        orb_minutes = self.config['trading_flags'].get('orb_minutes', 30)
        
        current_time = day_df.index[index].time()
        market_open_time = datetime.time(9, 15)
        orb_end_time = (datetime.datetime.combine(datetime.date.today(), market_open_time) + datetime.timedelta(minutes=orb_minutes)).time()
        
        if not self.orb_period_set and current_time >= orb_end_time:
            orb_df = day_df.between_time(market_open_time.strftime("%H:%M"), orb_end_time.strftime("%H:%M"))
            if not orb_df.empty:
                self.orb_high, self.orb_low = orb_df['high'].max(), orb_df['low'].min()
                self.orb_period_set = True
                logging.info(f"[{self.name}] ORB Set: High={self.orb_high:.2f}, Low={self.orb_low:.2f}, Range={(self.orb_high - self.orb_low):.2f}")
        
        if not self.orb_period_set: return 'HOLD'
        
        if (self.orb_high - self.orb_low) < 10:
            logging.debug(f"[{self.name}] ORB range is too narrow ({self.orb_high - self.orb_low:.2f} points). No trades will be taken.")
            return 'HOLD'

        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        if 'volume_ma' not in day_df.columns: day_df['volume_ma'] = day_df['volume'].rolling(window=20).mean()
        
        if sentiment in ['Bullish', 'Very Bullish'] and last['close'] < self.orb_high and current['close'] > self.orb_high and current['volume'] > (current.get('volume_ma', 0) * 1.5):
            logging.info(f"[{self.name}] BUY Signal on ORB High breakout.")
            return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and last['close'] > self.orb_low and current['close'] < self.orb_low and current['volume'] > (current.get('volume_ma', 0) * 1.5):
            logging.info(f"[{self.name}] SELL Signal on ORB Low breakdown.")
            return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if not self.orb_period_set:
            return f"Awaiting signal for {self.name}: Waiting for the opening range to be established."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price needs to break above the ORB high of {self.orb_high:.2f} on high volume."
        else:
            return f"Awaiting SELL signal: Price needs to break below the ORB low of {self.orb_low:.2f} on high volume."

class Bollinger_Band_Squeeze_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "BB_Squeeze_Breakout"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'

        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        if current['bb_bandwidth'] < current['bb_bandwidth_ma']:
            if sentiment in ['Bullish', 'Very Bullish'] and last['close'] < last['bb_upper'] and current['close'] > current['bb_upper']: return 'BUY'
            if sentiment in ['Bearish', 'Very Bearish'] and last['close'] > last['bb_lower'] and current['close'] < current['bb_lower']: return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        current = day_df.iloc[-1]
        if current['bb_bandwidth'] > current['bb_bandwidth_ma']:
            return f"Awaiting signal for {self.name}: Waiting for Bollinger Bands to tighten into a squeeze."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: In a BB Squeeze. Waiting for price to break above the upper band ({current['bb_upper']:.2f})."
        else:
            return f"Awaiting SELL signal: In a BB Squeeze. Waiting for price to break below the lower band ({current['bb_lower']:.2f})."

class MA_Crossover_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "MA_Crossover"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'

        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        if sentiment in ['Bullish', 'Very Bullish'] and last['ema_9'] <= last['ema_21'] and current['ema_9'] > current['ema_21']: return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and last['ema_9'] >= last['ema_21'] and current['ema_9'] < current['ema_21']: return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Waiting for the 9-period EMA to cross above the 21-period EMA."
        else:
            return f"Awaiting SELL signal: Waiting for the 9-period EMA to cross below the 21-period EMA."

class RSI_Divergence_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "RSI_Divergence"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        
        divergence = check_rsi_divergence(day_df.iloc[:index + 1], day_df['rsi'].iloc[:index + 1])
        if sentiment in ['Bullish', 'Very Bullish'] and divergence == 'Bullish': return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and divergence == 'Bearish': return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Waiting for price to make a new low while RSI makes a higher low (Bullish Divergence)."
        else:
            return f"Awaiting SELL signal: Waiting for price to make a new high while RSI makes a lower high (Bearish Divergence)."

class EMACrossRSIStrategy(BaseStrategy):
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "EMA_Cross_RSI"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        """
        Generates a signal if the EMAs are in a trending state and a crossover
        has occurred within a recent lookback period.
        """
        if index is None:
            index = len(day_df) - 1
        
        # New configurable lookback period. Default to 5 candles if not set.
        lookback_period = self.config['trading_flags'].get('ema_cross_lookback', 5)

        if index < lookback_period + 1: # Ensure we have enough data for the lookback
            return 'HOLD'

        # Ensure indicators are present
        if 'ema_9' not in day_df.columns: day_df['ema_9'] = calculate_ema(day_df['close'], 9)
        if 'ema_15' not in day_df.columns: day_df['ema_15'] = calculate_ema(day_df['close'], 15)
        if 'rsi' not in day_df.columns: day_df['rsi'] = calculate_rsi(day_df['close'], 14)
        
        current_candle = day_df.iloc[index]

        # --- MODIFIED BULLISH (BUY) SIGNAL LOGIC ---
        # 1. Check current state: 9-EMA is above 15-EMA now.
        is_trending_up = current_candle['ema_9'] > current_candle['ema_15']
        # 2. Check confirmation conditions: RSI and price are favorable now.
        is_confirmed_up = current_candle['rsi'] > 50 and current_candle['close'] > current_candle['ema_9']
        
        if is_trending_up and is_confirmed_up:
            # 3. Verify a "Golden Cross" happened recently
            recent_golden_cross = False
            for i in range(index - lookback_period, index + 1):
                prev_candle = day_df.iloc[i - 1]
                signal_candle = day_df.iloc[i]
                if prev_candle['ema_9'] < prev_candle['ema_15'] and signal_candle['ema_9'] > signal_candle['ema_15']:
                    recent_golden_cross = True
                    break  # Found the recent cross, no need to look further
            
            if recent_golden_cross:
                logging.info(f"[{self.name}] BUY Signal: 9/15 EMA in bullish state post-crossover with RSI > 50.")
                return 'BUY'

        # --- MODIFIED BEARISH (SELL) SIGNAL LOGIC ---
        # 1. Check current state: 9-EMA is below 15-EMA now.
        is_trending_down = current_candle['ema_9'] < current_candle['ema_15']
        # 2. Check confirmation conditions: RSI and price are favorable now.
        is_confirmed_down = current_candle['rsi'] < 50 and current_candle['close'] < current_candle['ema_9']

        if is_trending_down and is_confirmed_down:
            # 3. Verify a "Death Cross" happened recently
            recent_death_cross = False
            for i in range(index - lookback_period, index + 1):
                prev_candle = day_df.iloc[i - 1]
                signal_candle = day_df.iloc[i]
                if prev_candle['ema_9'] > prev_candle['ema_15'] and signal_candle['ema_9'] < signal_candle['ema_15']:
                    recent_death_cross = True
                    break
            
            if recent_death_cross:
                logging.info(f"[{self.name}] SELL Signal: 9/15 EMA in bearish state post-crossover with RSI < 50.")
                return 'SELL'

        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Waiting for 9-EMA to cross above 15-EMA, with confirmation from RSI > 50."
        else:
            return f"Awaiting SELL signal: Waiting for 9-EMA to cross below 15-EMA, with confirmation from RSI < 50."


class Reversal_Detector_Strategy(BaseStrategy):
    """
    A robust strategy that trades reversals based on a confluence of signals:
    1. Pre-Condition: An overextended trend.
    2. Primary Signal: RSI momentum divergence.
    3. Confirmation: A break of price structure (close over/under a fast EMA).
    """
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Reversal_Detector"
        self.is_reversal_trade = True # This flag bypasses the daily sentiment check

    def _is_trend_overextended(self, day_df, lookback=20):
        """Quantitatively defines an overextended trend."""
        price_slice = day_df['close'][-lookback:]
        max_price, min_price = price_slice.max(), price_slice.min()
        current_price = price_slice.iloc[-1]
        rsi = day_df['rsi'].iloc[-1]
        
        # Check for overextended uptrend 
        if (current_price / min_price - 1) > 0.015 and rsi > 70:
            return "Uptrend"
        # Check for overextended downtrend 
        if (max_price / current_price - 1) > 0.015 and rsi < 30:
            return "Downtrend"
            
        return "None"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        trend_status = is_trend_overextended(day_df)
        if trend_status == "None":
            return 'HOLD'

        rsi_divergence = check_momentum_divergence(day_df['close'], day_df['rsi'])
        
        current_candle = day_df.iloc[-1]

        # Look for a Bearish Reversal signal
        if trend_status == "Uptrend" and rsi_divergence == "Bearish":
            if current_candle['close'] < current_candle['ema_9']:
                logging.info(f"[{self.name}] Bearish Reversal Signal: Overextended uptrend with RSI divergence confirmed by close below 9-EMA.")
                return 'SELL'

        # Look for a Bullish Reversal signal
        if trend_status == "Downtrend" and rsi_divergence == "Bullish":
            if current_candle['close'] > current_candle['ema_9']:
                logging.info(f"[{self.name}] Bullish Reversal Signal: Overextended downtrend with RSI divergence confirmed by close above 9-EMA.")
                return 'BUY'
        
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        trend_status = is_trend_overextended(day_df)
        if trend_status == "Uptrend":
            return f"Awaiting SELL signal: Overextended uptrend detected. Looking for bearish RSI divergence and a confirmation break below 9-EMA."
        if trend_status == "Downtrend":
            return f"Awaiting BUY signal: Overextended downtrend detected. Looking for bullish RSI divergence and a confirmation break above 9-EMA."
        return f"Awaiting signal for {self.name}: Waiting for a sustained, overextended trend to form."

def get_strategy(name, kite, config):
    """Factory function to get a strategy instance by name."""
    strategies = {
        "Gemini_Default": Gemini_Default_Strategy, 
        "Supertrend_MACD": Supertrend_MACD_Strategy,
        "Volatility_Cluster_Reversal": VolatilityClusterStrategy,
        "Volume_Spread_Analysis": VSA_Strategy, 
        "Momentum_VWAP_RSI": Momentum_VWAP_RSI_Strategy,
        "Breakout_Prev_Day_HL": Breakout_Prev_Day_HL_Strategy,
        "Opening_Range_Breakout": Opening_Range_Breakout_Strategy,
        "BB_Squeeze_Breakout": Bollinger_Band_Squeeze_Strategy,
        "MA_Crossover": MA_Crossover_Strategy, 
        "RSI_Divergence": RSI_Divergence_Strategy,
        "EMA_Cross_RSI": EMACrossRSIStrategy,
        "Reversal_Detector": Reversal_Detector_Strategy, # New strategy added
    }
    strategy_class = strategies.get(name)
    if not strategy_class: raise ValueError(f"Strategy '{name}' not found.")
    return strategy_class(kite, config)