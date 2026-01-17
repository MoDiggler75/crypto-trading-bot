"""
AI Model Trainer
================
Trains machine learning models on the data collected by ai_data_farmer.py

Supported models:
1. Random Forest - Good baseline, interpretable
2. XGBoost - Often best performance
3. Neural Network - Can capture complex patterns
4. Ensemble - Combines multiple models

Usage:
  python ai_model_trainer.py
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import pickle

sys.stdout.reconfigure(line_buffering=True)

print("\n" + "="*70)
print("AI MODEL TRAINER - Machine Learning for Trading Signals")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

DATA_DIR = Path("ai_training_data")
MODEL_DIR = Path("ai_models")
MODEL_DIR.mkdir(exist_ok=True)

# Training settings
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 42

# Target to predict
TARGET = 'future_up_10'  # Predict if price goes up in next 10 bars

# Features to use (from ai_data_farmer.py)
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
# DATA LOADING
# ============================================

def load_latest_data():
    """Load the most recent combined training data"""
    files = list(DATA_DIR.glob("*_combined_*.csv"))
    if not files:
        print("No training data found! Run ai_data_farmer.py first.")
        return None

    # Get most recent file
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"Loading: {latest_file}")

    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows")

    return df

def prepare_data(df):
    """Prepare data for training"""
    # Check for required columns
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return None, None, None, None

    if TARGET not in df.columns:
        print(f"Missing target: {TARGET}")
        return None, None, None, None

    # Get features and target
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Handle any remaining NaN
    X = X.fillna(0)
    y = y.fillna(0)

    # Convert to numpy
    X = X.values
    y = y.values.astype(int)

    # Time-based split (don't use random - this is time series!)
    split_idx = int(len(X) * (1 - TEST_SIZE))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class balance (train): {y_train.mean()*100:.1f}% positive")
    print(f"Class balance (test): {y_test.mean()*100:.1f}% positive")

    return X_train, X_test, y_train, y_test

# ============================================
# MODEL TRAINING
# ============================================

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        print("\n--- Training Random Forest ---")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test)
        }

        print(f"  Train Accuracy: {metrics['train_accuracy']*100:.2f}%")
        print(f"  Test Accuracy:  {metrics['test_accuracy']*100:.2f}%")
        print(f"  Precision:      {metrics['precision']*100:.2f}%")
        print(f"  Recall:         {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:       {metrics['f1']*100:.2f}%")

        # Feature importance
        importance = pd.DataFrame({
            'feature': FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n  Top 10 Features:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        return model, metrics, importance

    except ImportError:
        print("  sklearn not installed. Run: pip install scikit-learn")
        return None, None, None

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        print("\n--- Training XGBoost ---")

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test)
        }

        print(f"  Train Accuracy: {metrics['train_accuracy']*100:.2f}%")
        print(f"  Test Accuracy:  {metrics['test_accuracy']*100:.2f}%")
        print(f"  Precision:      {metrics['precision']*100:.2f}%")
        print(f"  Recall:         {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:       {metrics['f1']*100:.2f}%")

        # Feature importance
        importance = pd.DataFrame({
            'feature': FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n  Top 10 Features:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        return model, metrics, importance

    except ImportError:
        print("  xgboost not installed. Run: pip install xgboost")
        return None, None, None

def train_neural_network(X_train, y_train, X_test, y_test):
    """Train Neural Network classifier"""
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        print("\n--- Training Neural Network ---")

        # Scale features (important for neural networks)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=64,
            learning_rate='adaptive',
            max_iter=200,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1
        )

        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test)
        }

        print(f"  Train Accuracy: {metrics['train_accuracy']*100:.2f}%")
        print(f"  Test Accuracy:  {metrics['test_accuracy']*100:.2f}%")
        print(f"  Precision:      {metrics['precision']*100:.2f}%")
        print(f"  Recall:         {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:       {metrics['f1']*100:.2f}%")

        return (model, scaler), metrics, None

    except ImportError:
        print("  sklearn not installed. Run: pip install scikit-learn")
        return None, None, None

# ============================================
# MODEL SAVING/LOADING
# ============================================

def save_model(model, name, metrics, features):
    """Save trained model to disk"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = MODEL_DIR / f"{name}_{timestamp}.pkl"

    model_data = {
        'model': model,
        'metrics': metrics,
        'features': features,
        'target': TARGET,
        'timestamp': timestamp
    }

    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  Model saved: {filename}")
    return filename

def load_model(filename):
    """Load trained model from disk"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def get_latest_model(name='random_forest'):
    """Get the most recently trained model"""
    files = list(MODEL_DIR.glob(f"{name}_*.pkl"))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_signal(model, features_dict, scaler=None):
    """Make a prediction for a single data point"""
    # Prepare feature vector
    X = np.array([[features_dict.get(f, 0) for f in FEATURES]])

    if scaler:
        X = scaler.transform(X)

    # Get prediction and probability
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return {
        'prediction': int(pred),
        'probability_up': float(prob[1]),
        'probability_down': float(prob[0]),
        'confidence': float(max(prob))
    }

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Load data
    print("[STEP 1] Loading training data...")
    df = load_latest_data()

    if df is None:
        print("\nNo data available. Run ai_data_farmer.py first!")
        sys.exit(1)

    # Prepare data
    print("\n[STEP 2] Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    if X_train is None:
        print("\nFailed to prepare data!")
        sys.exit(1)

    # Train models
    print("\n[STEP 3] Training models...")

    results = {}

    # Random Forest
    rf_model, rf_metrics, rf_importance = train_random_forest(X_train, y_train, X_test, y_test)
    if rf_model:
        save_model(rf_model, 'random_forest', rf_metrics, FEATURES)
        results['random_forest'] = rf_metrics

    # XGBoost
    xgb_model, xgb_metrics, xgb_importance = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_model:
        save_model(xgb_model, 'xgboost', xgb_metrics, FEATURES)
        results['xgboost'] = xgb_metrics

    # Neural Network
    nn_result, nn_metrics, _ = train_neural_network(X_train, y_train, X_test, y_test)
    if nn_result:
        save_model(nn_result, 'neural_network', nn_metrics, FEATURES)
        results['neural_network'] = nn_metrics

    # Summary
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)

    print("\n--- Model Comparison ---")
    print(f"{'Model':<20} {'Test Acc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['test_accuracy']*100:>9.2f}% {metrics['precision']*100:>9.2f}% {metrics['recall']*100:>9.2f}% {metrics['f1']*100:>9.2f}%")

    # Best model
    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        print(f"\nBest model by F1 score: {best_model}")

    print(f"\nModels saved to: {MODEL_DIR.absolute()}")
    print("\nNext steps:")
    print("  1. Use ai_signal_filter.py to filter trading signals")
    print("  2. Integrate with simulation_bot.py for AI-enhanced trading")
    print("\n" + "="*70)
