"""
AI Signal Filter
================
Filters trading signals using trained ML models.
Integrates with simulation_bot.py to only take high-probability trades.

Usage:
  from ai_signal_filter import AISignalFilter

  filter = AISignalFilter()
  should_trade, confidence = filter.evaluate_signal(features)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# ============================================
# CONFIGURATION
# ============================================

MODEL_DIR = Path("ai_models")
MIN_CONFIDENCE = 0.60  # Only take trades with 60%+ confidence

# Feature list (must match training)
FEATURES = [
    'rsi', 'rsi_overbought', 'rsi_oversold',
    'macd', 'macd_signal', 'macd_hist', 'macd_cross_up', 'macd_cross_down',
    'bb_width', 'bb_position',
    'atr_percent',
    'stoch_k', 'stoch_d',
    'price_vs_ema_9', 'price_vs_ema_21', 'price_vs_ema_50', 'price_vs_ema_200',
    'ema_9_21_cross',
    'vol_ratio',
    'roc_5', 'roc_10', 'roc_20', 'williams_r', 'cci',
    'body_size', 'upper_wick', 'lower_wick', 'candle_direction', 'consecutive_bars',
    'price_change_1', 'price_change_5', 'price_change_10',
    'volatility_5', 'volatility_20',
    'distance_from_high_20', 'distance_from_low_20', 'price_range'
]

# ============================================
# TECHNICAL INDICATOR CALCULATIONS
# ============================================

def calculate_rsi(prices, period=14):
    """Calculate RSI from price series"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD components"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger(prices, period=20, std_dev=2):
    """Calculate Bollinger Band position"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = (upper - lower) / sma
    position = (prices - lower) / (upper - lower)
    return width, position

def calculate_stochastic(high, low, close, period=14):
    """Calculate Stochastic oscillator"""
    lowest = low.rolling(window=period).min()
    highest = high.rolling(window=period).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(window=3).mean()
    return k, d

def calculate_features_from_ohlcv(df):
    """
    Calculate all features from OHLCV dataframe.
    Returns the latest row's features as a dictionary.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    open_price = df['Open']
    volume = df['Volume']

    features = {}

    # RSI
    rsi = calculate_rsi(close)
    features['rsi'] = rsi.iloc[-1]
    features['rsi_overbought'] = 1 if features['rsi'] > 70 else 0
    features['rsi_oversold'] = 1 if features['rsi'] < 30 else 0

    # MACD
    macd, signal, hist = calculate_macd(close)
    features['macd'] = macd.iloc[-1]
    features['macd_signal'] = signal.iloc[-1]
    features['macd_hist'] = hist.iloc[-1]
    features['macd_cross_up'] = 1 if (macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]) else 0
    features['macd_cross_down'] = 1 if (macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]) else 0

    # Bollinger
    bb_width, bb_position = calculate_bollinger(close)
    features['bb_width'] = bb_width.iloc[-1]
    features['bb_position'] = bb_position.iloc[-1]

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    features['atr_percent'] = (atr.iloc[-1] / close.iloc[-1]) * 100

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    features['stoch_k'] = stoch_k.iloc[-1]
    features['stoch_d'] = stoch_d.iloc[-1]

    # EMAs
    for period in [9, 21, 50, 200]:
        ema = close.ewm(span=period, adjust=False).mean()
        features[f'price_vs_ema_{period}'] = ((close.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]) * 100

    # EMA crossover
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    features['ema_9_21_cross'] = 1 if (ema_9.iloc[-1] > ema_21.iloc[-1] and ema_9.iloc[-2] <= ema_21.iloc[-2]) else 0

    # Volume
    vol_sma = volume.rolling(window=20).mean()
    features['vol_ratio'] = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

    # Momentum
    features['roc_5'] = ((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100 if len(close) > 5 else 0
    features['roc_10'] = ((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11]) * 100 if len(close) > 10 else 0
    features['roc_20'] = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100 if len(close) > 20 else 0

    # Williams %R
    highest = high.rolling(window=14).max()
    lowest = low.rolling(window=14).min()
    features['williams_r'] = -100 * (highest.iloc[-1] - close.iloc[-1]) / (highest.iloc[-1] - lowest.iloc[-1]) if highest.iloc[-1] != lowest.iloc[-1] else -50

    # CCI
    typical = (high + low + close) / 3
    sma_tp = typical.rolling(window=20).mean()
    mean_dev = typical.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    features['cci'] = (typical.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mean_dev.iloc[-1]) if mean_dev.iloc[-1] > 0 else 0

    # Candle patterns
    body = abs(close.iloc[-1] - open_price.iloc[-1])
    features['body_size'] = (body / open_price.iloc[-1]) * 100
    features['upper_wick'] = ((high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])) / open_price.iloc[-1]) * 100
    features['lower_wick'] = ((min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]) / open_price.iloc[-1]) * 100
    features['candle_direction'] = 1 if close.iloc[-1] > open_price.iloc[-1] else 0

    # Consecutive bars
    direction = (close > open_price).astype(int)
    current_dir = direction.iloc[-1]
    count = 1
    for i in range(2, min(len(direction), 20)):
        if direction.iloc[-i] == current_dir:
            count += 1
        else:
            break
    features['consecutive_bars'] = count

    # Price changes
    features['price_change_1'] = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100 if len(close) > 1 else 0
    features['price_change_5'] = ((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100 if len(close) > 5 else 0
    features['price_change_10'] = ((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11]) * 100 if len(close) > 10 else 0

    # Volatility
    features['volatility_5'] = (close.rolling(5).std() / close.rolling(5).mean() * 100).iloc[-1]
    features['volatility_20'] = (close.rolling(20).std() / close.rolling(20).mean() * 100).iloc[-1]

    # Range features
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    features['distance_from_high_20'] = ((close.iloc[-1] - high_20.iloc[-1]) / close.iloc[-1]) * 100
    features['distance_from_low_20'] = ((close.iloc[-1] - low_20.iloc[-1]) / close.iloc[-1]) * 100
    features['price_range'] = ((high_20.iloc[-1] - low_20.iloc[-1]) / close.iloc[-1]) * 100

    # Handle NaN
    for key in features:
        if pd.isna(features[key]):
            features[key] = 0

    return features

# ============================================
# AI SIGNAL FILTER CLASS
# ============================================

class AISignalFilter:
    """
    AI-based signal filter for trading decisions.
    """

    def __init__(self, model_name='random_forest', min_confidence=MIN_CONFIDENCE):
        self.model = None
        self.scaler = None
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.metrics = None
        self.loaded = False

        self._load_model()

    def _load_model(self):
        """Load the trained model"""
        # Find latest model file
        files = list(MODEL_DIR.glob(f"{self.model_name}_*.pkl"))
        if not files:
            print(f"[AI] No {self.model_name} model found. Train one first!")
            return

        latest = max(files, key=lambda x: x.stat().st_mtime)
        print(f"[AI] Loading model: {latest.name}")

        try:
            with open(latest, 'rb') as f:
                data = pickle.load(f)

            # Handle neural network (has scaler)
            if isinstance(data['model'], tuple):
                self.model, self.scaler = data['model']
            else:
                self.model = data['model']

            self.metrics = data['metrics']
            self.loaded = True

            print(f"[AI] Model loaded! Test accuracy: {self.metrics['test_accuracy']*100:.1f}%")

        except Exception as e:
            print(f"[AI] Error loading model: {e}")

    def evaluate_signal(self, features_dict, trade_type='LONG'):
        """
        Evaluate a trading signal.

        Args:
            features_dict: Dictionary of feature values
            trade_type: 'LONG' or 'SHORT'

        Returns:
            should_trade (bool): Whether to take the trade
            confidence (float): Model confidence (0-1)
            prediction (dict): Full prediction details
        """
        if not self.loaded:
            # No model - allow all trades
            return True, 0.5, {'error': 'No model loaded'}

        try:
            # Build feature vector
            X = np.array([[features_dict.get(f, 0) for f in FEATURES]])

            # Scale if needed (neural network)
            if self.scaler:
                X = self.scaler.transform(X)

            # Predict
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]

            prob_up = proba[1]
            prob_down = proba[0]

            # For LONG: we want high prob_up
            # For SHORT: we want high prob_down
            if trade_type == 'LONG':
                confidence = prob_up
                should_take = pred == 1 and confidence >= self.min_confidence
            else:  # SHORT
                confidence = prob_down
                should_take = pred == 0 and confidence >= self.min_confidence

            prediction = {
                'prediction': int(pred),
                'prob_up': float(prob_up),
                'prob_down': float(prob_down),
                'confidence': float(confidence),
                'trade_type': trade_type,
                'recommended': should_take
            }

            return should_take, confidence, prediction

        except Exception as e:
            print(f"[AI] Prediction error: {e}")
            return True, 0.5, {'error': str(e)}

    def evaluate_from_ohlcv(self, df, trade_type='LONG'):
        """
        Evaluate signal directly from OHLCV dataframe.

        Args:
            df: DataFrame with OHLCV data (at least 200 rows recommended)
            trade_type: 'LONG' or 'SHORT'

        Returns:
            Same as evaluate_signal()
        """
        if len(df) < 50:
            return True, 0.5, {'error': 'Insufficient data'}

        features = calculate_features_from_ohlcv(df)
        return self.evaluate_signal(features, trade_type)

    def get_status(self):
        """Get filter status"""
        return {
            'loaded': self.loaded,
            'model_name': self.model_name,
            'min_confidence': self.min_confidence,
            'metrics': self.metrics
        }

# ============================================
# STANDALONE USAGE
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AI SIGNAL FILTER - Testing")
    print("="*70 + "\n")

    # Test loading
    filter = AISignalFilter(model_name='random_forest')

    if not filter.loaded:
        print("\nNo model available. Train one first:")
        print("  1. Run: python ai_data_farmer.py")
        print("  2. Run: python ai_model_trainer.py")
        print("  3. Run: python ai_signal_filter.py (this script)")
    else:
        # Test with dummy features
        print("\nTesting with sample features...")

        dummy_features = {
            'rsi': 45,
            'macd_hist': 0.5,
            'bb_position': 0.5,
            'vol_ratio': 1.2,
            'price_change_5': 2.0
        }

        # Fill missing features with defaults
        for f in FEATURES:
            if f not in dummy_features:
                dummy_features[f] = 0

        should_trade, confidence, pred = filter.evaluate_signal(dummy_features, 'LONG')

        print(f"\nLONG Signal Evaluation:")
        print(f"  Should trade: {should_trade}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  Prob Up: {pred['prob_up']*100:.1f}%")
        print(f"  Prob Down: {pred['prob_down']*100:.1f}%")

        should_trade, confidence, pred = filter.evaluate_signal(dummy_features, 'SHORT')

        print(f"\nSHORT Signal Evaluation:")
        print(f"  Should trade: {should_trade}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  Prob Up: {pred['prob_up']*100:.1f}%")
        print(f"  Prob Down: {pred['prob_down']*100:.1f}%")

    print("\n" + "="*70)
