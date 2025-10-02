import pandas_ta as ta
import pandas as pd

def calculate_all_indicators(df: pd.DataFrame, config: dict):
    """
    Calculates and attaches all required technical indicators to the dataframe.
    This is called once per day to avoid redundant calculations.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

    # Gemini Default & RSI Divergence Strategy Indicators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_15'] = ta.ema(df['close'], length=15)
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_50'] = ta.ema(df['close'], length=50)

    # Supertrend & MACD Strategy Indicators
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
    if supertrend is not None and not supertrend.empty:
        df['supertrend_direction'] = supertrend['SUPERTd_10_3.0']
        
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']

    # Volatility & VSA Strategy Indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_ma'] = df['atr'].rolling(window=20).mean()
    df['spread'] = df['high'] - df['low']
    df['volume_ma'] = df['volume'].rolling(window=20).mean()

    # Momentum VWAP & ORB Strategy Indicators
    # This calculation will now work correctly
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

    # Bollinger Band Squeeze Strategy Indicators
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_mid'] = bbands['BBM_20_2.0']
        df['bb_bandwidth'] = bbands['BBB_20_2.0']
        df['bb_bandwidth_ma'] = df['bb_bandwidth'].rolling(window=20).mean()

    df.fillna(0, inplace=True) # Clean up any NaNs produced by indicators
    return df
