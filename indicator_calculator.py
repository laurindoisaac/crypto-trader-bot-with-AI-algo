import fin_primitives.pandas  # noqa: F401  registers .fp (pandas-ta method names)
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
    df['rsi'] = df['close'].fp.rsi(length=14)
    df['ema_9'] = df['close'].fp.ema(length=9)
    df['ema_15'] = df['close'].fp.ema(length=15)
    df['ema_20'] = df['close'].fp.ema(length=20)
    df['ema_21'] = df['close'].fp.ema(length=21)
    df['ema_50'] = df['close'].fp.ema(length=50)

    # Supertrend & MACD Strategy Indicators
    supertrend = df.fp.supertrend(length=10, multiplier=3)
    if supertrend is not None and not supertrend.empty:
        df['supertrend_direction'] = supertrend['SUPERTd_10_3.0']
        
    macd = df['close'].fp.macd(fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']

    # Volatility & VSA Strategy Indicators
    df['atr'] = df.fp.atr(length=14)
    df['atr_ma'] = df['atr'].rolling(window=20).mean()
    df['spread'] = df['high'] - df['low']
    df['volume_ma'] = df['volume'].rolling(window=20).mean()

    # Momentum VWAP & ORB Strategy Indicators
    # This calculation will now work correctly
    df['vwap'] = df.fp.vwap()

    # Bollinger Band Squeeze Strategy Indicators
    bbands = df['close'].fp.bbands(length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_mid'] = bbands['BBM_20_2.0']
        df['bb_bandwidth'] = bbands['BBB_20_2.0']
        df['bb_bandwidth_ma'] = df['bb_bandwidth'].rolling(window=20).mean()

    df.fillna(0, inplace=True) # Clean up any NaNs produced by indicators
    return df
