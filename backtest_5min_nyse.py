#!/usr/bin/env python3
"""
BACKTEST: 5-Minute NYSE Opening Range Breakout Strategy
Starting at 9:00 AM ET / 7:00 AM Arizona Time
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import pytz

print("\n" + "="*70)
print("5-MINUTE NYSE OPENING RANGE BREAKOUT STRATEGY")
print("Starting at 9:00 AM ET (7:00 AM Arizona Time)")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

starting_balance = 10000.0
max_concurrent_positions = 5
risk_per_trade_pct = 0.05

# Safety limits
CIRCUIT_BREAKER_LOSS = starting_balance * 0.20
TRAILING_STOP_PCT = 0.15
DAILY_LOSS_LIMIT = starting_balance * 0.05
MAX_POSITION_VALUE = starting_balance * 0.30
MAX_LOSS_PER_TRADE = 200.0

# Fees
FEE_PCT = 0.002
SLIPPAGE_PCT = 0.0005

# Strategy
SL_BUFFER = 0.03

# NYSE Opening Time (9:00 AM ET = 7:00 AM Arizona MST)
# Arizona is UTC-7 (MST, no DST)
# NYSE is UTC-5 (EST) or UTC-4 (EDT)
# For simplicity, we'll use 7:00 AM Arizona = 14:00 UTC during winter
NYSE_OPEN_HOUR_AZ = 7  # 7:00 AM Arizona time
NYSE_OPEN_MINUTE_AZ = 0

print(f"Starting Balance: ${starting_balance:,.2f}")
print(f"Max Positions: {max_concurrent_positions}")
print(f"Risk per Trade: {risk_per_trade_pct*100}%")
print(f"NYSE Open Time (Arizona): {NYSE_OPEN_HOUR_AZ:02d}:{NYSE_OPEN_MINUTE_AZ:02d}")

# ============================================
# LOAD DATA
# ============================================

print(f"\n[STEP 1] Loading Kraken 5-minute data...\n")

crypto_dir = "Kraken_OHLCVT"

# Tokens to test
TARGET_TOKENS = [
    'ETH', 'SOL', 'SUI', 'LINK', 'DOGE', 'XRP', 'ADA', 'AVAX', 'LTC', 'PEPE',
    'XMR', 'DOT', 'ALGO', 'XLM', 'BCH', 'XPL', 'TAO', 'ZEC', 'BAT', 'POL',
    'TRX', 'AAVE', 'QNT', 'ATOM', 'COMP', 'HNT', 'APE', 'ACH', 'CAKE', 'LQTY',
    'BOND', 'KNC', 'ANKR', 'ALCX', 'OMG', 'SC', 'UNI', 'HBAR', 'BNB'
]

# Find pairs with 5-min data
pairs_5 = set([os.path.basename(f).replace('_5.csv', '')
               for f in glob.glob(f'{crypto_dir}/*USD_5.csv')])

# Filter to only target tokens
valid_pairs = []
for pair in pairs_5:
    token = pair.replace('USD', '')
    if token in TARGET_TOKENS:
        valid_pairs.append(pair)

valid_pairs = sorted(valid_pairs)

print(f"Found {len(valid_pairs)} pairs with 5-minute data")

# Load 5-minute data
m5_data = {}

for pair_name in valid_pairs:
    m5_path = os.path.join(crypto_dir, f"{pair_name}_5.csv")
    if os.path.exists(m5_path):
        try:
            df_m5 = pd.read_csv(m5_path, header=None,
                               names=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades'])
            df_m5['datetime'] = pd.to_datetime(df_m5['timestamp'], unit='s', utc=True)

            # Convert to Arizona time (MST = UTC-7, no DST)
            arizona_tz = pytz.timezone('America/Phoenix')
            df_m5['datetime_az'] = df_m5['datetime'].dt.tz_convert(arizona_tz)

            df_m5['date'] = df_m5['datetime_az'].dt.date
            df_m5['hour'] = df_m5['datetime_az'].dt.hour
            df_m5['minute'] = df_m5['datetime_az'].dt.minute

            if len(df_m5) > 0:
                m5_data[pair_name] = df_m5
                date_range = f"{df_m5['datetime_az'].min().date()} to {df_m5['datetime_az'].max().date()}"
                print(f"  [OK] {pair_name}: {len(df_m5):,} candles ({date_range})")
        except Exception as e:
            print(f"  [ERROR] {pair_name}: Error loading - {e}")

valid_pairs = [p for p in valid_pairs if p in m5_data]
print(f"\n[SUCCESS] Loaded {len(valid_pairs)} pairs")

# ============================================
# TECHNICAL INDICATORS
# ============================================

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def check_volume_spike(current_volume, recent_volumes, threshold=1.5):
    avg_volume = np.mean(recent_volumes)
    if avg_volume == 0:
        return False, 0

    volume_ratio = current_volume / avg_volume
    return volume_ratio >= threshold, volume_ratio

# ============================================
# GET TRADING DATES
# ============================================

print(f"\n[STEP 2] Finding trading dates...\n")

all_dates = set()
for pair in valid_pairs:
    dates = set(m5_data[pair]['date'].unique())
    all_dates.update(dates)

all_dates = sorted(list(all_dates))
print(f"Trading period: {all_dates[0]} to {all_dates[-1]}")
print(f"Total days: {len(all_dates)}")

# ============================================
# BACKTEST
# ============================================

print(f"\n[STEP 3] Running 5-minute NYSE Opening Range backtest...\n")

account_balance = starting_balance
peak_balance = starting_balance

open_positions = {}
closed_trades = []

total_signals = 0
trades_executed = 0

circuit_breaker_triggered = False
current_day_pnl = 0
previous_date = None

csv_file = 'backtest_5min_nyse_results.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Trade#', 'Pair', 'Date', 'Type', 'Entry', 'Exit', 'Reason',
                    'Quantity', 'PnL', 'Balance', 'Peak', 'Drawdown%', 'RSI', 'VolRatio'])

for date_idx, date_val in enumerate(all_dates):

    if (date_idx + 1) % max(1, len(all_dates) // 20) == 0:
        progress = (date_idx + 1) / len(all_dates) * 100
        print(f"  {progress:.0f}% - Day {date_idx + 1}/{len(all_dates)} - "
              f"Balance: ${account_balance:,.2f} - Trades: {trades_executed}")

    # Safety checks
    total_pnl = account_balance - starting_balance
    if total_pnl <= -CIRCUIT_BREAKER_LOSS:
        print(f"\n[CIRCUIT BREAKER] Loss: ${total_pnl:.2f}")
        circuit_breaker_triggered = True
        break

    if account_balance > peak_balance:
        peak_balance = account_balance

    drawdown = (peak_balance - account_balance) / peak_balance
    if drawdown >= TRAILING_STOP_PCT:
        print(f"\n[TRAILING STOP] Drawdown: {drawdown*100:.1f}%")
        circuit_breaker_triggered = True
        break

    # Daily loss limit
    if previous_date != date_val:
        current_day_pnl = 0
        previous_date = date_val

    if current_day_pnl <= -DAILY_LOSS_LIMIT:
        continue

    # Get 5-minute candles for today
    m5_candles = {}
    opening_candles = {}

    for pair in valid_pairs:
        day_m5 = m5_data[pair][m5_data[pair]['date'] == date_val]
        if len(day_m5) > 0:
            m5_candles[pair] = day_m5

            # Find the first candle at or after NYSE open time (7:00 AM AZ)
            opening_candle = day_m5[
                (day_m5['hour'] == NYSE_OPEN_HOUR_AZ) &
                (day_m5['minute'] == NYSE_OPEN_MINUTE_AZ)
            ]

            if len(opening_candle) > 0:
                opening_candles[pair] = opening_candle.iloc[0]

    if not opening_candles:
        continue

    # ============================================
    # CHECK EXITS
    # ============================================

    pairs_to_close = []

    for pair in list(open_positions.keys()):
        if pair not in m5_candles:
            continue

        position = open_positions[pair]

        for idx, candle in m5_candles[pair].iterrows():
            # Only check candles after the opening candle
            if candle['datetime_az'] <= position['entry_time']:
                continue

            exit_price = None
            exit_reason = None

            if position['type'] == 'LONG':
                if candle['low'] <= position['sl']:
                    exit_price = position['sl'] * (1 - SLIPPAGE_PCT)
                    exit_reason = 'SL'
                elif candle['high'] >= position['tp']:
                    exit_price = position['tp'] * (1 - SLIPPAGE_PCT)
                    exit_reason = 'TP'
            else:
                if candle['high'] >= position['sl']:
                    exit_price = position['sl'] * (1 + SLIPPAGE_PCT)
                    exit_reason = 'SL'
                elif candle['low'] <= position['tp']:
                    exit_price = position['tp'] * (1 + SLIPPAGE_PCT)
                    exit_reason = 'TP'

            if exit_price:
                if position['type'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']

                total_fees = (position['entry_price'] * position['quantity'] * FEE_PCT) + \
                            (exit_price * position['quantity'] * FEE_PCT)
                pnl -= total_fees

                account_balance += pnl
                current_day_pnl += pnl

                if account_balance > peak_balance:
                    peak_balance = account_balance

                drawdown_pct = ((peak_balance - account_balance) / peak_balance * 100)

                closed_trades.append({
                    'pair': pair,
                    'date': str(date_val),
                    'type': position['type'],
                    'entry': position['entry_price'],
                    'exit': exit_price,
                    'exit_reason': exit_reason,
                    'quantity': position['quantity'],
                    'pnl': pnl
                })

                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        len(closed_trades), pair, str(date_val), position['type'],
                        f"{position['entry_price']:.8f}", f"{exit_price:.8f}",
                        exit_reason, f"{position['quantity']:.6f}",
                        f"{pnl:.2f}", f"{account_balance:.2f}",
                        f"{peak_balance:.2f}", f"{drawdown_pct:.2f}",
                        f"{position.get('rsi', 0):.1f}",
                        f"{position.get('vol_ratio', 0):.2f}"
                    ])

                pairs_to_close.append(pair)
                break

    for pair in pairs_to_close:
        del open_positions[pair]

    # ============================================
    # CHECK ENTRIES
    # ============================================

    for pair in valid_pairs:
        if pair in open_positions or len(open_positions) >= max_concurrent_positions:
            continue

        if pair not in opening_candles or pair not in m5_candles:
            continue

        # Get the opening range from the 7:00 AM candle
        opening_candle = opening_candles[pair]
        orb_high = opening_candle['high']
        orb_low = opening_candle['low']
        opening_time = opening_candle['datetime_az']

        if orb_high == orb_low:
            continue

        # Get candles after the opening candle
        candles_after_open = m5_candles[pair][m5_candles[pair]['datetime_az'] > opening_time]

        if len(candles_after_open) == 0:
            continue

        # Get historical data for indicators (before opening candle)
        all_m5_before = m5_data[pair][m5_data[pair]['datetime_az'] <= opening_time]

        if len(all_m5_before) < 20:
            continue

        recent_volumes = all_m5_before['volume'].tail(20).values

        # Check each 5-minute candle after opening for breakout
        for idx, m5_candle in candles_after_open.iterrows():
            close_price = m5_candle['close']
            signal_type = None

            if close_price > orb_high:
                signal_type = "LONG"
            elif close_price < orb_low:
                signal_type = "SHORT"

            if signal_type:
                total_signals += 1

                # Calculate RSI for tracking (not filtering)
                recent_closes = all_m5_before['close'].tail(20).values.tolist()
                recent_closes.append(close_price)
                rsi = calculate_rsi(recent_closes, period=14)

                # Volume calculation for tracking (not filtering)
                has_volume, vol_ratio = check_volume_spike(
                    m5_candle['volume'],
                    recent_volumes,
                    1.5
                )

                # Calculate position
                entry_price = close_price

                if signal_type == "LONG":
                    entry_price = entry_price * (1 + SLIPPAGE_PCT)
                    stop_loss = orb_low - (orb_low * SL_BUFFER)
                    risk_per_unit = entry_price - stop_loss
                    if risk_per_unit <= 0:
                        continue
                    take_profit = entry_price + (2 * risk_per_unit)
                else:
                    entry_price = entry_price * (1 - SLIPPAGE_PCT)
                    stop_loss = orb_high + (orb_high * SL_BUFFER)
                    risk_per_unit = stop_loss - entry_price
                    if risk_per_unit <= 0:
                        continue
                    take_profit = entry_price - (2 * risk_per_unit)

                # Position sizing
                risk_amount = account_balance * risk_per_trade_pct
                quantity = risk_amount / risk_per_unit

                # Safety caps
                position_value = quantity * entry_price
                if position_value > MAX_POSITION_VALUE:
                    quantity = MAX_POSITION_VALUE / entry_price

                # Dynamic max loss per trade (scales with balance)
                max_loss_dynamic = min(account_balance * 0.02, MAX_LOSS_PER_TRADE)
                potential_loss = quantity * risk_per_unit
                if potential_loss > max_loss_dynamic:
                    quantity = max_loss_dynamic / risk_per_unit

                # Enter position
                open_positions[pair] = {
                    'type': signal_type,
                    'entry_price': entry_price,
                    'entry_time': m5_candle['datetime_az'],
                    'sl': stop_loss,
                    'tp': take_profit,
                    'quantity': quantity,
                    'rsi': rsi if rsi else 0,
                    'vol_ratio': vol_ratio
                }
                trades_executed += 1
                break

# ============================================
# RESULTS
# ============================================

print(f"\n" + "="*70)
print("5-MINUTE NYSE OPENING RANGE BACKTEST RESULTS")
print("="*70 + "\n")

total_pnl = account_balance - starting_balance
roi = (total_pnl / starting_balance * 100)
winning_trades = sum(1 for t in closed_trades if t['pnl'] > 0)
losing_trades = sum(1 for t in closed_trades if t['pnl'] < 0)

print("Account Summary:")
print(f"  Starting: ${starting_balance:,.2f}")
print(f"  Peak: ${peak_balance:,.2f}")
print(f"  Final: ${account_balance:,.2f}")
print(f"  P&L: ${total_pnl:+,.2f}")
print(f"  ROI: {roi:+.2f}%")

max_dd = ((peak_balance - account_balance) / peak_balance * 100) if peak_balance > 0 else 0
print(f"  Max Drawdown: {max_dd:.2f}%\n")

print("Signal Quality:")
print(f"  Total signals: {total_signals}")
print(f"  Executed: {trades_executed}\n")

print("Trade Statistics:")
print(f"  Closed trades: {len(closed_trades)}")

if closed_trades:
    win_rate = (winning_trades / len(closed_trades) * 100)
    print(f"  Win rate: {win_rate:.1f}%")

    wins = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in closed_trades if t['pnl'] < 0]

    if wins:
        print(f"  Average win: ${sum(wins)/len(wins):.2f}")
        print(f"  Largest win: ${max(wins):.2f}")
    if losses:
        print(f"  Average loss: ${sum(losses)/len(losses):.2f}")
        print(f"  Largest loss: ${min(losses):.2f}")

    if wins and losses:
        profit_factor = abs(sum(wins)) / abs(sum(losses))
        print(f"  Profit factor: {profit_factor:.2f}")

print(f"\n[SUCCESS] Results saved to {csv_file}")
print("="*70 + "\n")
