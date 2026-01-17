"""
AI Data Farmer Bot
==================
This bot collects and processes market data for machine learning training.
It does NOT execute trades - it focuses on:
1. Collecting OHLCV data from multiple sources
2. Computing technical indicators
3. Labeling data with outcomes (for supervised learning)
4. Building feature sets for ML models
5. Storing processed data for training

Run this to build your training dataset before training AI models.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("\n" + "="*70)
print("AI DATA FARMER - Market Data Collection & Processing")
print("MODE: DATA COLLECTION (No trading)")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

# Data storage
DATA_DIR = Path("ai_training_data")
DATA_DIR.mkdir(exist_ok=True)

# Symbols to collect
CRYPTO_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
    'AVAX-USD', 'LINK-USD', 'DOT-USD', 'LTC-USD', 'ATOM-USD'
]

# Timeframes to collect
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']

# Lookback period for data collection
LOOKBACK_DAYS = 365  # 1 year of data

# Technical indicator settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
STOCH_PERIOD = 14
EMA_PERIODS = [9, 21, 50, 200]

# ============================================
# TECHNICAL INDICATORS
# ============================================

def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD, Signal line, and Histogram"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=BB_PERIOD, std_dev=BB_STD):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_width = (upper_band - lower_band) / sma
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return upper_band, sma, lower_band, bb_width, bb_position

def calculate_atr(high, low, close, period=ATR_PERIOD):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_stochastic(high, low, close, period=STOCH_PERIOD):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_volume_indicators(df):
    """Calculate volume-based indicators"""
    # Volume SMA
    vol_sma = df['Volume'].rolling(window=20).mean()
    vol_ratio = df['Volume'] / vol_sma

    # On-Balance Volume (OBV)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Volume-Price Trend
    vpt = ((df['Close'].diff() / df['Close'].shift()) * df['Volume']).fillna(0).cumsum()

    return vol_sma, vol_ratio, obv, vpt

def calculate_momentum_indicators(df):
    """Calculate momentum indicators"""
    # Rate of Change
    roc_5 = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    roc_10 = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    roc_20 = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100

    # Williams %R
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)

    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mean_dev = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma_tp) / (0.015 * mean_dev)

    return roc_5, roc_10, roc_20, williams_r, cci

def calculate_pattern_features(df):
    """Calculate candlestick pattern features"""
    # Candle body size (normalized)
    body_size = abs(df['Close'] - df['Open']) / df['Open'] * 100

    # Upper/lower wick sizes
    upper_wick = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open'] * 100
    lower_wick = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open'] * 100

    # Candle direction
    direction = (df['Close'] > df['Open']).astype(int)

    # Consecutive same direction candles
    direction_change = direction != direction.shift()
    consecutive_count = direction_change.cumsum()
    consecutive_bars = direction.groupby(consecutive_count).cumcount() + 1

    return body_size, upper_wick, lower_wick, direction, consecutive_bars

# ============================================
# FEATURE ENGINEERING
# ============================================

def add_all_features(df):
    """Add all technical indicators and features to dataframe"""
    print("    Computing technical indicators...")

    # Price data
    close = df['Close']
    high = df['High']
    low = df['Low']

    # RSI
    df['rsi'] = calculate_rsi(close)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(close)
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) &
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) &
                              (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

    # Bollinger Bands
    df['bb_upper'], df['bb_mid'], df['bb_lower'], df['bb_width'], df['bb_position'] = calculate_bollinger_bands(close)

    # ATR
    df['atr'] = calculate_atr(high, low, close)
    df['atr_percent'] = df['atr'] / close * 100

    # Stochastic
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(high, low, close)

    # EMAs
    for period in EMA_PERIODS:
        df[f'ema_{period}'] = calculate_ema(close, period)
        df[f'price_vs_ema_{period}'] = (close - df[f'ema_{period}']) / df[f'ema_{period}'] * 100

    # EMA crossovers
    df['ema_9_21_cross'] = ((df['ema_9'] > df['ema_21']) &
                            (df['ema_9'].shift(1) <= df['ema_21'].shift(1))).astype(int)

    # Volume indicators
    df['vol_sma'], df['vol_ratio'], df['obv'], df['vpt'] = calculate_volume_indicators(df)

    # Momentum indicators
    df['roc_5'], df['roc_10'], df['roc_20'], df['williams_r'], df['cci'] = calculate_momentum_indicators(df)

    # Pattern features
    df['body_size'], df['upper_wick'], df['lower_wick'], df['candle_direction'], df['consecutive_bars'] = calculate_pattern_features(df)

    # Price change features
    df['price_change_1'] = close.pct_change(1) * 100
    df['price_change_5'] = close.pct_change(5) * 100
    df['price_change_10'] = close.pct_change(10) * 100

    # Volatility features
    df['volatility_5'] = close.rolling(5).std() / close.rolling(5).mean() * 100
    df['volatility_20'] = close.rolling(20).std() / close.rolling(20).mean() * 100

    # High/Low features
    df['distance_from_high_20'] = (close - high.rolling(20).max()) / close * 100
    df['distance_from_low_20'] = (close - low.rolling(20).min()) / close * 100

    # Support/Resistance levels (simple)
    df['recent_high'] = high.rolling(20).max()
    df['recent_low'] = low.rolling(20).min()
    df['price_range'] = (df['recent_high'] - df['recent_low']) / close * 100

    print(f"    Added {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features")

    return df

# ============================================
# DATA LABELING
# ============================================

def add_labels(df, forward_periods=[5, 10, 20]):
    """Add target labels for supervised learning"""
    print("    Creating labels...")

    close = df['Close']

    for period in forward_periods:
        # Future return (what we're trying to predict)
        df[f'future_return_{period}'] = close.shift(-period).pct_change(period) * 100

        # Binary classification: up or down
        df[f'future_up_{period}'] = (close.shift(-period) > close).astype(int)

        # Multi-class: strong down, down, neutral, up, strong up
        returns = (close.shift(-period) - close) / close * 100
        df[f'future_class_{period}'] = pd.cut(
            returns,
            bins=[-np.inf, -2, -0.5, 0.5, 2, np.inf],
            labels=[0, 1, 2, 3, 4]  # strong_down, down, neutral, up, strong_up
        )

        # Optimal trade outcome (for RL)
        # Did price go up by 2% before going down by 1%?
        future_highs = df['High'].rolling(window=period).max().shift(-period)
        future_lows = df['Low'].rolling(window=period).min().shift(-period)

        max_gain = (future_highs - close) / close * 100
        max_loss = (close - future_lows) / close * 100

        df[f'max_gain_{period}'] = max_gain
        df[f'max_loss_{period}'] = max_loss

        # Reward-to-risk ratio
        df[f'reward_risk_{period}'] = max_gain / (max_loss + 0.001)

    return df

# ============================================
# DATA COLLECTION
# ============================================

def download_yahoo_data(symbol, period='1y', interval='1h'):
    """Download data from Yahoo Finance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if len(df) > 0:
            return df
    except Exception as e:
        print(f"    Error downloading {symbol}: {e}")
    return None

def download_kraken_data(pair, interval=60):
    """Download data from Kraken API"""
    try:
        import krakenex
        from pykrakenapi import KrakenAPI

        api = krakenex.API()
        k = KrakenAPI(api)

        ohlc, last = k.get_ohlc_data(pair, interval=interval, ascending=True)
        return ohlc
    except Exception as e:
        print(f"    Error downloading from Kraken: {e}")
    return None

# ============================================
# MAIN DATA FARMING LOOP
# ============================================

def farm_data_yahoo(symbols, timeframe='1h', lookback='1y'):
    """Farm data from Yahoo Finance"""
    all_data = {}

    for symbol in symbols:
        print(f"  Downloading {symbol}...")
        df = download_yahoo_data(symbol, period=lookback, interval=timeframe)

        if df is not None and len(df) > 100:
            # Add features
            df = add_all_features(df)

            # Add labels
            df = add_labels(df)

            # Drop NaN rows (from indicator calculations)
            df = df.dropna()

            all_data[symbol] = df
            print(f"    {len(df)} rows with {len(df.columns)} columns")
        else:
            print(f"    Skipped - insufficient data")

        time.sleep(0.5)  # Rate limiting

    return all_data

def save_training_data(all_data, prefix='training'):
    """Save processed data to files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save individual symbol files
    for symbol, df in all_data.items():
        filename = DATA_DIR / f"{prefix}_{symbol.replace('-', '_')}_{timestamp}.csv"
        df.to_csv(filename)
        print(f"  Saved {filename}")

    # Save combined dataset
    if all_data:
        combined = pd.concat([
            df.assign(symbol=symbol)
            for symbol, df in all_data.items()
        ])
        combined_file = DATA_DIR / f"{prefix}_combined_{timestamp}.csv"
        combined.to_csv(combined_file)
        print(f"\n  Combined dataset: {combined_file}")
        print(f"  Total rows: {len(combined)}")

    return timestamp

def load_training_data(filename):
    """Load processed training data"""
    df = pd.read_csv(DATA_DIR / filename, index_col=0, parse_dates=True)
    return df

# ============================================
# DATA STATISTICS & ANALYSIS
# ============================================

def analyze_dataset(df, symbol='Unknown'):
    """Analyze the dataset and print statistics"""
    print(f"\n--- Dataset Analysis: {symbol} ---")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total rows: {len(df)}")
    print(f"Total features: {len(df.columns)}")

    # Feature categories
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    indicator_cols = [c for c in df.columns if c not in price_cols and 'future' not in c]
    label_cols = [c for c in df.columns if 'future' in c]

    print(f"\nPrice columns: {len(price_cols)}")
    print(f"Indicator columns: {len(indicator_cols)}")
    print(f"Label columns: {len(label_cols)}")

    # Label distribution
    if 'future_up_10' in df.columns:
        up_pct = df['future_up_10'].mean() * 100
        print(f"\nLabel distribution (10-bar future):")
        print(f"  Up: {up_pct:.1f}%")
        print(f"  Down: {100-up_pct:.1f}%")

    # Feature statistics
    print(f"\nKey indicator ranges:")
    for col in ['rsi', 'macd_hist', 'bb_position', 'atr_percent', 'vol_ratio']:
        if col in df.columns:
            print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

def get_feature_list():
    """Return list of features for ML training"""
    features = [
        # RSI
        'rsi', 'rsi_overbought', 'rsi_oversold',

        # MACD
        'macd', 'macd_signal', 'macd_hist', 'macd_cross_up', 'macd_cross_down',

        # Bollinger Bands
        'bb_width', 'bb_position',

        # ATR
        'atr_percent',

        # Stochastic
        'stoch_k', 'stoch_d',

        # EMAs
        'price_vs_ema_9', 'price_vs_ema_21', 'price_vs_ema_50', 'price_vs_ema_200',
        'ema_9_21_cross',

        # Volume
        'vol_ratio',

        # Momentum
        'roc_5', 'roc_10', 'roc_20', 'williams_r', 'cci',

        # Patterns
        'body_size', 'upper_wick', 'lower_wick', 'candle_direction', 'consecutive_bars',

        # Price changes
        'price_change_1', 'price_change_5', 'price_change_10',

        # Volatility
        'volatility_5', 'volatility_20',

        # Range
        'distance_from_high_20', 'distance_from_low_20', 'price_range'
    ]
    return features

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("[STEP 1] Starting data collection from Yahoo Finance...")
    print(f"Symbols: {len(CRYPTO_SYMBOLS)}")
    print(f"Timeframe: 1 hour")
    print(f"Lookback: {LOOKBACK_DAYS} days\n")

    # Collect data
    print("Downloading and processing data...")
    all_data = farm_data_yahoo(CRYPTO_SYMBOLS, timeframe='1h', lookback='1y')

    if not all_data:
        print("\nNo data collected! Check your internet connection.")
        sys.exit(1)

    # Save data
    print("\n[STEP 2] Saving training data...")
    timestamp = save_training_data(all_data, prefix='crypto_1h')

    # Analyze
    print("\n[STEP 3] Analyzing datasets...")
    for symbol, df in all_data.items():
        analyze_dataset(df, symbol)

    # Summary
    print("\n" + "="*70)
    print("DATA FARMING COMPLETE")
    print("="*70)
    print(f"\nFiles saved to: {DATA_DIR.absolute()}")
    print(f"Timestamp: {timestamp}")
    print(f"\nFeatures available for training: {len(get_feature_list())}")
    print("\nNext steps:")
    print("  1. Run ai_model_trainer.py to train ML models")
    print("  2. Use trained model in ai_signal_filter.py")
    print("  3. Integrate with simulation_bot.py for AI-filtered trading")
    print("\n" + "="*70)
