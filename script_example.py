#!/usr/bin/env python3
"""
ORB Trading Simulator
Uses 240-minute (4-hour) Breaking Zone + 5-minute breakout detection
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN

# Import pandas and pytz for CSV loading
import pandas as pd
import pytz

# ============================================
# ORB TRADING BOT SIMULATOR
# Backtesting with 4-hour BZ and 5-minute breakouts
# ============================================

print("\n" + "="*70)
print("ORB TRADING BOT SIMULATOR")
print("4-hour Breaking Zone + 5-minute Breakout Detection")
print("="*70 + "\n")

# ============================================
# STEP 1: CONFIGURATION
# ============================================

print("[STEP 1] Configuration\n")

# Hardcoded configuration
starting_balance = 20000.0
leverage = 10.0
max_concurrent_positions = 5
max_risk_pct = 5.0 / 100  # 5% risk per trade

# Calculate per-trade amount based on leverage
available_leverage = starting_balance * leverage
risk_per_trade = available_leverage / max_concurrent_positions
max_risk_allowed = starting_balance * max_risk_pct

print(f"\n[OK] Configuration set")
print(f"  Capital: ${starting_balance:,.2f}")
print(f"  Leverage: {leverage}x")
print(f"  Buying power: ${available_leverage:,.2f}")
print(f"  Max positions: {max_concurrent_positions}")
print(f"  Per trade amount: ${risk_per_trade:,.2f}")
print(f"  Max risk per trade: ${max_risk_allowed:,.2f} ({max_risk_pct*100:.1f}%)")

# ============================================
# STEP 2: SELECT TRADING PAIRS
# ============================================

print("\n[STEP 2] Select trading pairs\n")

# Auto-discover all USD pairs with both 4-hour and 5-min data
import glob
import random
crypto_dir_temp = "Kraken_OHLCVT"
pairs_240 = set([os.path.basename(f).replace('USD_240.csv', '') for f in glob.glob(f'{crypto_dir_temp}/*USD_240.csv')])
pairs_5 = set([os.path.basename(f).replace('USD_5.csv', '') for f in glob.glob(f'{crypto_dir_temp}/*USD_5.csv')])
all_pairs = sorted(list(pairs_240.intersection(pairs_5)))

# Pick 5 random pairs
random.seed()  # Use current time for randomness
TRADING_PAIRS = random.sample(all_pairs, min(5, len(all_pairs)))

print(f"Auto-discovered {len(all_pairs)} total pairs")
print(f"Randomly selected 5 pairs to test: {', '.join(TRADING_PAIRS)}")

# ============================================
# STEP 3: LOAD CSV DATA (4-HOUR AND 5-MINUTE)
# ============================================

print("\n[STEP 3] Loading data...\n")

crypto_dir = "Kraken_OHLCVT"

if not os.path.exists(crypto_dir):
    print(f"[ERROR] Folder not found: {crypto_dir}")
    exit()

# Load 4-hour data for Breaking Zone
bz_data = {}
m5_data = {}

for pair in TRADING_PAIRS:
    # Try to find files: ETHUSD_240.csv and ETHUSD_5.csv
    bz_file = f"{pair}USD_240.csv"
    m5_file = f"{pair}USD_5.csv"
    
    bz_path = os.path.join(crypto_dir, bz_file)
    m5_path = os.path.join(crypto_dir, m5_file)  # 5-min files are in same directory
    
    # Load 4-hour data
    if os.path.exists(bz_path):
        try:
            df_bz = pd.read_csv(bz_path, header=None)
            num_cols = len(df_bz.columns)
            if num_cols == 7:
                df_bz.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'flags']
            elif num_cols == 6:
                df_bz.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            else:
                continue
            
            df_bz['timestamp'] = pd.to_datetime(df_bz['timestamp'], unit='s', utc=True)
            ny_tz = pytz.timezone('America/New_York')
            df_bz['timestamp'] = df_bz['timestamp'].dt.tz_convert(ny_tz)
            df_bz['date'] = df_bz['timestamp'].dt.strftime('%Y-%m-%d')
            
            bz_data[pair] = {
                'df': df_bz,
                'candles': df_bz.to_dict('records')
            }
            print(f"  [OK] {pair} 4-hour: {len(df_bz):,} candles")
        except Exception as e:
            print(f"  [ERROR] {pair} 4-hour failed: {e}")
    else:
        print(f"  [ERROR] {bz_file} not found")
    
    # Load 5-minute data
    if os.path.exists(m5_path):
        try:
            df_m5 = pd.read_csv(m5_path, header=None)
            num_cols = len(df_m5.columns)
            if num_cols == 7:
                df_m5.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'flags']
            elif num_cols == 6:
                df_m5.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            else:
                continue
            
            df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], unit='s', utc=True)
            ny_tz = pytz.timezone('America/New_York')
            df_m5['timestamp'] = df_m5['timestamp'].dt.tz_convert(ny_tz)
            df_m5['date'] = df_m5['timestamp'].dt.strftime('%Y-%m-%d')
            
            m5_data[pair] = {
                'df': df_m5,
                'candles': df_m5.to_dict('records')
            }
            print(f"  [OK] {pair} 5-min: {len(df_m5):,} candles")
        except Exception as e:
            print(f"  [ERROR] {pair} 5-min failed: {e}")

# Verify we have both for each pair
valid_pairs = [p for p in TRADING_PAIRS if p in bz_data and p in m5_data]

if not valid_pairs:
    print("\n[ERROR] No pairs have both 4-hour and 5-minute data!")
    exit()

print(f"\n[OK] Valid pairs for trading: {', '.join(valid_pairs)}")

# ============================================
# STEP 4: SL BUFFERS
# ============================================

print("\n[STEP 4] Loading SL buffers...\n")

SL_BUFFER_LOOKUP = {
    'XBT': 0.001,
    'ETH': 0.001,
    'SOL': 0.001,
    'ADA': 0.001,
    'XRP': 0.001,
    'DOT': 0.001,
    'LTC': 0.001,
    'DOGE': 0.001,
    'SUI': 0.001,
    'LINK': 0.001,
    'ALGO': 0.001,
    'BCH': 0.001,
    'BNB': 0.001,
    'UNI': 0.001,
    'HBAR': 0.001,
    'AAVE': 0.001,
    'TRX': 0.001
}

print(f"[OK] SL Buffers configured\n")

# ============================================
# STEP 5: GET COMMON DATE RANGE
# ============================================

all_dates = set()
for pair in valid_pairs:
    dates = set(m5_data[pair]['df']['date'].unique())
    all_dates.update(dates)

all_dates = sorted(list(all_dates))

print(f"[STEP 5] Trading period: {all_dates[0]} to {all_dates[-1]}")
print(f"Total trading days: {len(all_dates)}\n")

# ============================================
# STEP 6: BACKTESTING LOOP
# ============================================

print(f"[STEP 6] Simulating trading...\n")

open_positions = {}
closed_trades = []
account_balance = starting_balance
signals_detected = 0
trades_executed = 0

for date_idx, date_str in enumerate(all_dates):
    if (date_idx + 1) % max(1, len(all_dates) // 10) == 0:
        progress = (date_idx + 1) / len(all_dates) * 100
        print(f"  {progress:.0f}% - Day {date_idx + 1}/{len(all_dates)} - Balance: ${account_balance:,.2f} - Trades: {trades_executed}")
    
    # Get 4-hour candle for this day (Breaking Zone)
    # Use ONLY the FIRST 4-hour candle to establish the breakout zone
    bz_candles = {}
    for pair in valid_pairs:
        bz_daily = [c for c in bz_data[pair]['candles'] if c['date'] == date_str]
        if bz_daily:
            # Use FIRST 4-hour candle of the day as the Breaking Zone
            bz_candles[pair] = bz_daily[0]
    
    # Get 5-minute candles for this day (Breakout detection)
    m5_candles = {}
    for pair in valid_pairs:
        m5_daily = [c for c in m5_data[pair]['candles'] if c['date'] == date_str]
        if m5_daily:
            m5_candles[pair] = m5_daily
    
    if not m5_candles or not bz_candles:
        continue
    
    # Check exits first
    pairs_to_close = []
    for pair in list(open_positions.keys()):
        if pair not in m5_candles:
            continue
        
        position = open_positions[pair]
        
        for candle in m5_candles[pair]:
            candle_high = candle['high']
            candle_low = candle['low']
            
            exit_price = None
            exit_reason = None
            
            if position['type'] == 'LONG':
                if candle_low <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif candle_high >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            else:  # SHORT
                if candle_high >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL'
                elif candle_low <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TP'
            
            if exit_price:
                pnl = position['tp_pnl'] if exit_reason == 'TP' else position['sl_pnl']
                
                closed_trades.append({
                    'pair': pair,
                    'date': date_str,
                    'type': position['type'],
                    'entry': position['entry_price'],
                    'exit': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl
                })
                
                account_balance += pnl
                pairs_to_close.append(pair)
                break  # Exit this pair
    
    # Close positions
    for pair in pairs_to_close:
        del open_positions[pair]
    
    # Check entries - use 4-hour BZ
    for pair in valid_pairs:
        if pair in open_positions or len(open_positions) >= max_concurrent_positions:
            continue
        
        if pair not in bz_candles or pair not in m5_candles:
            continue
        
        bz_candle = bz_candles[pair]
        bz_high = bz_candle['high']
        bz_low = bz_candle['low']
        
        # Check 5-minute candles for breakout
        for m5_candle in m5_candles[pair]:
            close = m5_candle['close']
            signal_type = None
            
            if close > bz_high:
                signal_type = "LONG"
            elif close < bz_low:
                signal_type = "SHORT"
            
            if signal_type:
                signals_detected += 1
                
                # Recalculate position sizing (dynamic)
                current_available_leverage = account_balance * leverage
                current_risk_per_trade = current_available_leverage / max_concurrent_positions
                current_max_risk_allowed = account_balance * max_risk_pct

                # Calculate SL/TP
                sl_buffer = SL_BUFFER_LOOKUP.get(pair, 0.001)
                entry_price = close
                
                if signal_type == "LONG":
                    stop_loss = bz_low - (bz_low * sl_buffer)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        continue
                    take_profit = entry_price + (2 * risk)
                else:
                    stop_loss = bz_high + (bz_high * sl_buffer)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        continue
                    take_profit = entry_price - (2 * risk)
                
                # Calculate position size based on risk (not full buying power)
                # Position size = (max dollar risk / price risk per unit)
                position_value = (current_max_risk_allowed / risk) * entry_price

                # But cap it at available buying power per trade
                position_value = min(position_value, current_risk_per_trade)

                # Calculate quantity
                usd_dec = Decimal(str(position_value))
                price_dec = Decimal(str(entry_price))
                quantity = float((usd_dec / price_dec).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
                
                # Track position
                open_positions[pair] = {
                    'type': signal_type,
                    'entry_price': entry_price,
                    'sl': stop_loss,
                    'tp': take_profit,
                    'quantity': quantity,
                    'tp_pnl': (take_profit - entry_price) * quantity if signal_type == "LONG" else (entry_price - take_profit) * quantity,
                    'sl_pnl': (stop_loss - entry_price) * quantity if signal_type == "LONG" else (entry_price - stop_loss) * quantity
                }
                trades_executed += 1
                break  # One trade per pair per day

# ============================================
# RESULTS
# ============================================

print(f"\n" + "="*70)
print("TRADING RESULTS")
print("="*70 + "\n")

total_pnl = sum(t['pnl'] for t in closed_trades)
roi = (total_pnl / starting_balance * 100) if starting_balance else 0
winning_trades = sum(1 for t in closed_trades if t['pnl'] > 0)
losing_trades = sum(1 for t in closed_trades if t['pnl'] < 0)

print("Account Summary:")
print(f"  Starting balance: ${starting_balance:,.2f}")
print(f"  Final balance: ${account_balance:,.2f}")
print(f"  Total P&L: ${total_pnl:+,.2f}")
print(f"  ROI: {roi:+.2f}%\n")

print("Trade Statistics:")
print(f"  Total closed trades: {len(closed_trades)}")
if closed_trades:
    print(f"  Winning trades: {winning_trades} ({(winning_trades/len(closed_trades)*100):.1f}%)")
    print(f"  Losing trades: {losing_trades} ({(losing_trades/len(closed_trades)*100):.1f}%)")
    
    wins = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in closed_trades if t['pnl'] < 0]
    if wins:
        print(f"  Average win: ${sum(wins)/len(wins):+.2f}")
    if losses:
        print(f"  Average loss: ${sum(losses)/len(losses):+.2f}")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'starting_balance': starting_balance,
        'leverage': leverage,
        'max_positions': max_concurrent_positions,
        'max_risk_pct': max_risk_pct * 100
    },
    'results': {
        'final_balance': account_balance,
        'total_pnl': total_pnl,
        'roi': roi,
        'total_trades': len(closed_trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades
    },
    'trades': closed_trades
}

with open('live_trading_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('live_trading_log.csv', 'w') as f:
    f.write("Pair,Date,Type,Entry,Exit,Reason,P&L\n")
    for trade in closed_trades:
        f.write(f"{trade['pair']},{trade['date']},{trade['type']},")
        f.write(f"{trade['entry']:.8f},{trade['exit']:.8f},{trade['exit_reason']},")
        f.write(f"{trade['pnl']:+.2f}\n")

print(f"\n[OK] Results saved to live_trading_results.json and live_trading_log.csv")
print(f"\n" + "="*70 + "\n")