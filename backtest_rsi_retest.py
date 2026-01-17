#!/usr/bin/env python3
"""
BACKTEST: RSI Zone Retest Strategy
Using Yahoo Finance Data

Strategy:
- Track when RSI enters extreme zones (below 30 or above 70)
- Wait for RSI to EXIT the zone (return to neutral)
- Enter trade when RSI RE-ENTERS the zone (confirmation)

LONG Signal (oversold retest):
1. RSI drops below 30 (enters oversold zone)
2. RSI rises back above 30 (exits zone)
3. RSI drops below 30 again (re-enters zone) → ENTER LONG

SHORT Signal (overbought retest):
1. RSI rises above 70 (enters overbought zone)
2. RSI drops back below 70 (exits zone)
3. RSI rises above 70 again (re-enters zone) → ENTER SHORT
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pytz

print("\n" + "="*70)
print("RSI ZONE RETEST STRATEGY - YAHOO FINANCE")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

starting_balance = 10000.0
max_concurrent_positions = 5
risk_per_trade_pct = 0.05

# Safety limits
CIRCUIT_BREAKER_LOSS = starting_balance * 0.20
MAX_POSITION_VALUE = starting_balance * 0.30

# Fees
FEE_PCT = 0.001
SLIPPAGE_PCT = 0.0005

# Strategy
SL_BUFFER = 0.02  # 2% stop loss from entry
COOLDOWN_MINUTES = 30  # Longer cooldown for this strategy

# RSI Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30   # Below this = oversold zone
RSI_OVERBOUGHT = 70  # Above this = overbought zone

# NYSE Times
NYSE_OPEN_HOUR = 9
NYSE_OPEN_MINUTE = 30
NYSE_CLOSE_HOUR = 16

print(f"Starting Balance: ${starting_balance:,.2f}")
print(f"Max Positions: {max_concurrent_positions}")
print(f"Risk per Trade: {risk_per_trade_pct*100}%")
print(f"Stop Loss: {SL_BUFFER*100}%")
print(f"Take Profit: {SL_BUFFER*2*100}% (2:1 R/R)")
print(f"\nRSI Retest Strategy:")
print(f"  RSI Period: {RSI_PERIOD}")
print(f"  Oversold Zone: RSI < {RSI_OVERSOLD}")
print(f"  Overbought Zone: RSI > {RSI_OVERBOUGHT}")
print(f"  Signal: Enter on RETEST of zone (second entry)")

# ============================================
# SYMBOLS TO TEST
# ============================================

STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD',
    'JPM', 'V', 'MA', 'UNH', 'JNJ', 'PG', 'HD', 'DIS',
    'NFLX', 'CRM', 'PYPL', 'SHOP', 'ROKU', 'UBER', 'ABNB',
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'ARKK'
]

CRYPTO_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
    'AVAX-USD', 'LINK-USD', 'DOT-USD', 'LTC-USD',
    'ATOM-USD', 'XLM-USD', 'ALGO-USD', 'AAVE-USD'
]

USE_CRYPTO = False
symbols_to_test = CRYPTO_SYMBOLS if USE_CRYPTO else STOCK_SYMBOLS

print(f"\nTesting {len(symbols_to_test)} symbols")

# ============================================
# RSI CALCULATION
# ============================================

def calculate_rsi(prices, period=14):
    """Calculate RSI for a price series"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))

    if len(gains) >= period:
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 100)

    return rsi

# ============================================
# DOWNLOAD DATA
# ============================================

print(f"\n[STEP 1] Downloading 5-minute data from Yahoo Finance...")

all_data = {}
period = "60d"
interval = "5m"

for symbol in symbols_to_test:
    try:
        print(f"  Downloading {symbol}...", end=" ")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if len(df) > 0:
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')

            df['RSI'] = calculate_rsi(df['Close'].values, RSI_PERIOD)
            all_data[symbol] = df
            print(f"{len(df)} candles")
        else:
            print("No data")
    except Exception as e:
        print(f"Error: {e}")

print(f"\nLoaded data for {len(all_data)} symbols")

if len(all_data) == 0:
    print("ERROR: No data loaded. Exiting.")
    exit(1)

# ============================================
# BACKTEST ENGINE
# ============================================

print(f"\n[STEP 2] Running backtest...")

# State
account_balance = starting_balance
peak_balance = starting_balance
open_positions = {}
closed_trades = []
daily_traded_pairs = {}
pair_cooldowns = {}

# Track zone state for each symbol
# States: 'neutral', 'in_oversold', 'exited_oversold', 'in_overbought', 'exited_overbought'
zone_states = {symbol: 'neutral' for symbol in all_data.keys()}

# Get all unique dates
all_dates = set()
for symbol, df in all_data.items():
    all_dates.update(df.index.date)

all_dates = sorted(all_dates)
print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
print(f"Total trading days: {len(all_dates)}")

# Counters for analysis
patterns_detected = {'oversold_retest': 0, 'overbought_retest': 0}
signals_taken = {'long': 0, 'short': 0}

# Process each day
for current_date in all_dates:
    daily_traded_pairs[current_date] = set()

    # Close overnight positions (safety)
    for symbol in list(open_positions.keys()):
        del open_positions[symbol]

    # Reset zone states at start of each day
    for symbol in zone_states:
        zone_states[symbol] = 'neutral'

    # Process each symbol
    for symbol, df in all_data.items():
        day_data = df[df.index.date == current_date]

        if len(day_data) < RSI_PERIOD + 1:
            continue

        # Only trade during market hours
        trading_data = day_data[
            (day_data.index.hour >= NYSE_OPEN_HOUR) &
            (day_data.index.hour < 15) |
            ((day_data.index.hour == 15) & (day_data.index.minute <= 45))
        ]

        prev_rsi = None

        for idx, candle in trading_data.iterrows():
            current_time = idx
            close_price = candle['Close']
            high_price = candle['High']
            low_price = candle['Low']
            rsi = candle['RSI']

            if rsi == 0 or np.isnan(rsi):
                continue

            current_state = zone_states[symbol]

            # ============================================
            # CHECK EXITS FOR OPEN POSITIONS
            # ============================================
            if symbol in open_positions:
                pos = open_positions[symbol]
                exit_price = None
                exit_reason = None

                # Check stop loss
                if pos['type'] == 'LONG' and low_price <= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'STOP_LOSS'
                elif pos['type'] == 'SHORT' and high_price >= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'STOP_LOSS'

                # Check take profit
                if pos['type'] == 'LONG' and high_price >= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TAKE_PROFIT'
                elif pos['type'] == 'SHORT' and low_price <= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TAKE_PROFIT'

                if exit_price:
                    exit_price = exit_price * (1 - SLIPPAGE_PCT) if pos['type'] == 'LONG' else exit_price * (1 + SLIPPAGE_PCT)
                    fee = exit_price * pos['quantity'] * FEE_PCT

                    if pos['type'] == 'LONG':
                        pnl = (exit_price - pos['entry_price']) * pos['quantity'] - fee
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['quantity'] - fee

                    account_balance += pnl
                    if account_balance > peak_balance:
                        peak_balance = account_balance

                    pair_cooldowns[symbol] = current_time + timedelta(minutes=COOLDOWN_MINUTES)

                    closed_trades.append({
                        'symbol': symbol,
                        'type': pos['type'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'entry_rsi': pos['entry_rsi'],
                        'exit_rsi': rsi,
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'balance': account_balance,
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'date': str(current_date)
                    })

                    del open_positions[symbol]
                    # Reset zone state after trade
                    zone_states[symbol] = 'neutral'
                    continue

            # ============================================
            # ZONE STATE MACHINE
            # ============================================

            # Track RSI zone transitions
            in_oversold = rsi < RSI_OVERSOLD
            in_overbought = rsi > RSI_OVERBOUGHT

            signal_type = None

            if current_state == 'neutral':
                # First entry into a zone
                if in_oversold:
                    zone_states[symbol] = 'in_oversold'
                elif in_overbought:
                    zone_states[symbol] = 'in_overbought'

            elif current_state == 'in_oversold':
                # Check if RSI exited oversold zone
                if not in_oversold:
                    zone_states[symbol] = 'exited_oversold'

            elif current_state == 'exited_oversold':
                # Check if RSI re-entered oversold zone (RETEST)
                if in_oversold:
                    # LONG SIGNAL - Oversold retest confirmed!
                    patterns_detected['oversold_retest'] += 1
                    signal_type = 'LONG'
                    zone_states[symbol] = 'neutral'  # Reset after signal
                elif in_overbought:
                    # Price moved to overbought, reset
                    zone_states[symbol] = 'in_overbought'

            elif current_state == 'in_overbought':
                # Check if RSI exited overbought zone
                if not in_overbought:
                    zone_states[symbol] = 'exited_overbought'

            elif current_state == 'exited_overbought':
                # Check if RSI re-entered overbought zone (RETEST)
                if in_overbought:
                    # SHORT SIGNAL - Overbought retest confirmed!
                    patterns_detected['overbought_retest'] += 1
                    signal_type = 'SHORT'
                    zone_states[symbol] = 'neutral'  # Reset after signal
                elif in_oversold:
                    # Price moved to oversold, reset
                    zone_states[symbol] = 'in_oversold'

            # ============================================
            # ENTER TRADE ON CONFIRMED SIGNAL
            # ============================================

            if signal_type:
                # Check if we can take the trade
                if symbol in open_positions:
                    continue
                if len(open_positions) >= max_concurrent_positions:
                    continue
                if symbol in daily_traded_pairs.get(current_date, set()):
                    continue
                if symbol in pair_cooldowns and current_time < pair_cooldowns[symbol]:
                    continue

                # Calculate position
                entry_price = close_price * (1 + SLIPPAGE_PCT) if signal_type == 'LONG' else close_price * (1 - SLIPPAGE_PCT)

                if signal_type == 'LONG':
                    stop_loss = entry_price * (1 - SL_BUFFER)
                    risk_per_unit = entry_price - stop_loss
                    take_profit = entry_price + (2 * risk_per_unit)
                    signals_taken['long'] += 1
                else:
                    stop_loss = entry_price * (1 + SL_BUFFER)
                    risk_per_unit = stop_loss - entry_price
                    take_profit = entry_price - (2 * risk_per_unit)
                    signals_taken['short'] += 1

                risk_amount = account_balance * risk_per_trade_pct
                quantity = risk_amount / risk_per_unit

                position_value = quantity * entry_price
                if position_value > MAX_POSITION_VALUE:
                    quantity = MAX_POSITION_VALUE / entry_price

                fee = entry_price * quantity * FEE_PCT
                account_balance -= fee

                open_positions[symbol] = {
                    'type': signal_type,
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'entry_rsi': rsi,
                    'sl': stop_loss,
                    'tp': take_profit,
                    'quantity': quantity
                }

                daily_traded_pairs[current_date].add(symbol)

            prev_rsi = rsi

    # END OF DAY: Close all remaining positions
    for symbol in list(open_positions.keys()):
        pos = open_positions[symbol]

        if symbol in all_data:
            day_data = all_data[symbol][all_data[symbol].index.date == current_date]
            if len(day_data) > 0:
                last_candle = day_data.iloc[-1]
                exit_price = last_candle['Close']
                exit_rsi = last_candle['RSI']

                exit_price = exit_price * (1 - SLIPPAGE_PCT) if pos['type'] == 'LONG' else exit_price * (1 + SLIPPAGE_PCT)
                fee = exit_price * pos['quantity'] * FEE_PCT

                if pos['type'] == 'LONG':
                    pnl = (exit_price - pos['entry_price']) * pos['quantity'] - fee
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['quantity'] - fee

                account_balance += pnl
                if account_balance > peak_balance:
                    peak_balance = account_balance

                closed_trades.append({
                    'symbol': symbol,
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'exit_reason': 'END_OF_DAY',
                    'entry_rsi': pos['entry_rsi'],
                    'exit_rsi': exit_rsi,
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'balance': account_balance,
                    'entry_time': pos['entry_time'],
                    'exit_time': last_candle.name,
                    'date': str(current_date)
                })

        del open_positions[symbol]

# ============================================
# RESULTS
# ============================================

print("\n" + "="*70)
print("BACKTEST RESULTS - RSI ZONE RETEST STRATEGY")
print("="*70)

total_trades = len(closed_trades)

if total_trades == 0:
    print("\nNo trades executed!")
    print(f"Patterns detected: {patterns_detected}")
    exit(0)

winning_trades = [t for t in closed_trades if t['pnl'] > 0]
losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
total_pnl = account_balance - starting_balance
roi = (total_pnl / starting_balance) * 100

gross_profit = sum(t['pnl'] for t in winning_trades)
gross_loss = abs(sum(t['pnl'] for t in losing_trades))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

running_balance = starting_balance
peak = starting_balance
max_drawdown = 0
for trade in closed_trades:
    running_balance = trade['balance']
    if running_balance > peak:
        peak = running_balance
    drawdown = (peak - running_balance) / peak * 100
    if drawdown > max_drawdown:
        max_drawdown = drawdown

print(f"\nStarting Balance: ${starting_balance:,.2f}")
print(f"Final Balance:    ${account_balance:,.2f}")
print(f"Total P&L:        ${total_pnl:+,.2f}")
print(f"ROI:              {roi:+.2f}%")

print(f"\nTotal Trades:     {total_trades}")
print(f"Winning Trades:   {len(winning_trades)}")
print(f"Losing Trades:    {len(losing_trades)}")
print(f"Win Rate:         {win_rate:.1f}%")

print(f"\nProfit Factor:    {profit_factor:.2f}")
print(f"Max Drawdown:     {max_drawdown:.2f}%")

if winning_trades:
    avg_win = np.mean([t['pnl'] for t in winning_trades])
    print(f"Avg Win:          ${avg_win:.2f}")

if losing_trades:
    avg_loss = np.mean([t['pnl'] for t in losing_trades])
    print(f"Avg Loss:         ${avg_loss:.2f}")

# Pattern analysis
print(f"\n--- PATTERN ANALYSIS ---")
print(f"Oversold Retests Detected:  {patterns_detected['oversold_retest']}")
print(f"Overbought Retests Detected: {patterns_detected['overbought_retest']}")
print(f"LONG Signals Taken:  {signals_taken['long']}")
print(f"SHORT Signals Taken: {signals_taken['short']}")

# Exit breakdown
tp_exits = len([t for t in closed_trades if t.get('exit_reason') == 'TAKE_PROFIT'])
sl_exits = len([t for t in closed_trades if t.get('exit_reason') == 'STOP_LOSS'])
eod_exits = len([t for t in closed_trades if t.get('exit_reason') == 'END_OF_DAY'])
print(f"\nTake Profit Exits: {tp_exits}")
print(f"Stop Loss Exits:   {sl_exits}")
print(f"End of Day Exits:  {eod_exits}")

# Performance by direction
long_trades = [t for t in closed_trades if t['type'] == 'LONG']
short_trades = [t for t in closed_trades if t['type'] == 'SHORT']

if long_trades:
    long_pnl = sum(t['pnl'] for t in long_trades)
    long_wr = len([t for t in long_trades if t['pnl'] > 0]) / len(long_trades) * 100
    print(f"\nLONG Performance:  ${long_pnl:+.2f} ({len(long_trades)} trades, {long_wr:.1f}% WR)")

if short_trades:
    short_pnl = sum(t['pnl'] for t in short_trades)
    short_wr = len([t for t in short_trades if t['pnl'] > 0]) / len(short_trades) * 100
    print(f"SHORT Performance: ${short_pnl:+.2f} ({len(short_trades)} trades, {short_wr:.1f}% WR)")

# By symbol
print("\n" + "-"*50)
print("PERFORMANCE BY SYMBOL")
print("-"*50)

symbol_stats = {}
for trade in closed_trades:
    s = trade['symbol']
    if s not in symbol_stats:
        symbol_stats[s] = {'trades': 0, 'pnl': 0, 'wins': 0}
    symbol_stats[s]['trades'] += 1
    symbol_stats[s]['pnl'] += trade['pnl']
    if trade['pnl'] > 0:
        symbol_stats[s]['wins'] += 1

sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)

for symbol, stats in sorted_symbols[:15]:
    wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
    print(f"{symbol:12} | Trades: {stats['trades']:3} | P&L: ${stats['pnl']:+8.2f} | WR: {wr:5.1f}%")

# Save results
results = {
    'config': {
        'strategy': 'RSI Zone Retest',
        'starting_balance': starting_balance,
        'sl_buffer': SL_BUFFER,
        'risk_per_trade': risk_per_trade_pct,
        'rsi_period': RSI_PERIOD,
        'rsi_oversold': RSI_OVERSOLD,
        'rsi_overbought': RSI_OVERBOUGHT,
        'symbols': symbols_to_test
    },
    'results': {
        'final_balance': account_balance,
        'total_pnl': total_pnl,
        'roi': roi,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'patterns_detected': patterns_detected,
        'signals_taken': signals_taken
    },
    'trades': closed_trades
}

with open('backtest_rsi_retest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to backtest_rsi_retest_results.json")
print("\n" + "="*70)
