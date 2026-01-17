#!/usr/bin/env python3
"""
BACKTEST: 4-Hour Breaking Zone Strategy
Using Yahoo Finance Data

Strategy:
- First 4-hour candle of the day establishes the "breaking zone"
- Wait for 4-hour candle to COMPLETE before trading
- Use 5-minute candles for trade signals (breakout above/below zone)
- Close all positions at end of day (no overnight holds)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pytz

print("\n" + "="*70)
print("4-HOUR BREAKING ZONE BACKTEST - YAHOO FINANCE")
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

# Fees (estimated for typical broker)
FEE_PCT = 0.001  # 0.1% per trade
SLIPPAGE_PCT = 0.0005

# Strategy
SL_BUFFER = 0.02  # 2% stop loss from entry
COOLDOWN_MINUTES = 3

# NYSE Times
NYSE_OPEN_HOUR = 9
NYSE_OPEN_MINUTE = 30  # NYSE opens at 9:30 AM ET
NYSE_CLOSE_HOUR = 16   # NYSE closes at 4:00 PM ET

# 4-Hour Zone: First 4 hours = 9:30 AM to 1:30 PM ET
ZONE_END_HOUR = 13
ZONE_END_MINUTE = 30

print(f"Starting Balance: ${starting_balance:,.2f}")
print(f"Max Positions: {max_concurrent_positions}")
print(f"Risk per Trade: {risk_per_trade_pct*100}%")
print(f"Stop Loss: {SL_BUFFER*100}%")
print(f"Take Profit: {SL_BUFFER*2*100}% (2:1 R/R)")
print(f"\nStrategy: 4-Hour Breaking Zone (9:30 AM - 1:30 PM ET)")
print(f"Trading Window: 1:30 PM - 4:00 PM ET")
print(f"Positions closed at end of each day")

# ============================================
# SYMBOLS TO TEST
# ============================================

# Stock symbols to test
STOCK_SYMBOLS = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD',
    # Other Large Cap
    'JPM', 'V', 'MA', 'UNH', 'JNJ', 'PG', 'HD', 'DIS',
    # Growth/Momentum
    'NFLX', 'CRM', 'PYPL', 'SQ', 'SHOP', 'ROKU', 'UBER', 'ABNB',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'ARKK'
]

# Crypto symbols
CRYPTO_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
    'AVAX-USD', 'LINK-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD',
    'UNI-USD', 'ATOM-USD', 'XLM-USD', 'ALGO-USD', 'AAVE-USD'
]

# Choose which to use
USE_CRYPTO = False  # Changed to False for stocks
symbols_to_test = CRYPTO_SYMBOLS if USE_CRYPTO else STOCK_SYMBOLS

print(f"\nTesting {len(symbols_to_test)} symbols")
print(f"Symbols: {', '.join(symbols_to_test[:5])}...")

# ============================================
# DOWNLOAD DATA
# ============================================

print(f"\n[STEP 1] Downloading 5-minute data from Yahoo Finance...")
print("(This may take a minute...)\n")

# Yahoo Finance limits:
# - 5-minute data available for last 60 days only

all_data = {}
period = "60d"  # Max for 5-minute data
interval = "5m"

for symbol in symbols_to_test:
    try:
        print(f"  Downloading {symbol}...", end=" ")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if len(df) > 0:
            # Ensure timezone aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')

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
daily_traded_pairs = {}  # {date: set(symbols)}
pair_cooldowns = {}  # {symbol: datetime}

# Get all unique dates across all symbols
all_dates = set()
for symbol, df in all_data.items():
    dates = df.index.date
    all_dates.update(dates)

all_dates = sorted(all_dates)
print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
print(f"Total trading days: {len(all_dates)}")

# Process each day
for current_date in all_dates:
    # Reset daily state
    daily_traded_pairs[current_date] = set()

    # Close any positions from previous day (shouldn't happen, but safety check)
    for symbol in list(open_positions.keys()):
        pos = open_positions[symbol]
        # This shouldn't happen as we close at EOD, but just in case
        del open_positions[symbol]

    # Build 4-hour breaking zone for each symbol
    # Zone = High/Low of first 4 hours (9:30 AM - 1:30 PM ET)
    breaking_zones = {}

    for symbol, df in all_data.items():
        # Filter to current date
        day_data = df[df.index.date == current_date]

        if len(day_data) == 0:
            continue

        # Get all candles in the 4-hour zone (9:30 AM to 1:30 PM)
        zone_candles = day_data[
            (day_data.index.hour >= NYSE_OPEN_HOUR) &
            (
                (day_data.index.hour < ZONE_END_HOUR) |
                ((day_data.index.hour == ZONE_END_HOUR) & (day_data.index.minute < ZONE_END_MINUTE))
            )
        ]

        if len(zone_candles) < 10:  # Need enough candles to form a valid zone
            continue

        # Breaking zone = High and Low of the 4-hour period
        zone_high = zone_candles['High'].max()
        zone_low = zone_candles['Low'].min()
        zone_close_time = zone_candles.index[-1]

        breaking_zones[symbol] = {
            'high': zone_high,
            'low': zone_low,
            'close_time': zone_close_time,
            'candle_count': len(zone_candles)
        }

    if len(breaking_zones) == 0:
        continue

    # Process each 5-minute candle AFTER the 4-hour zone closes (1:30 PM onwards)
    for symbol, df in all_data.items():
        if symbol not in breaking_zones:
            continue

        bz = breaking_zones[symbol]
        day_data = df[df.index.date == current_date]

        # Only process candles AFTER the zone is complete (1:30 PM onwards)
        trading_candles = day_data[day_data.index > bz['close_time']]

        # Also filter to before market close (4:00 PM)
        trading_candles = trading_candles[trading_candles.index.hour < NYSE_CLOSE_HOUR]

        for idx, candle in trading_candles.iterrows():
            current_time = idx
            close_price = candle['Close']
            high_price = candle['High']
            low_price = candle['Low']

            # Check exits for open positions
            if symbol in open_positions:
                pos = open_positions[symbol]
                exit_price = None
                exit_reason = None

                if pos['type'] == 'LONG':
                    if low_price <= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'STOP_LOSS'
                    elif high_price >= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'TAKE_PROFIT'
                else:  # SHORT
                    if high_price >= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'STOP_LOSS'
                    elif low_price <= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'TAKE_PROFIT'

                if exit_price:
                    # Calculate P&L
                    exit_price = exit_price * (1 - SLIPPAGE_PCT) if pos['type'] == 'LONG' else exit_price * (1 + SLIPPAGE_PCT)
                    fee = exit_price * pos['quantity'] * FEE_PCT

                    if pos['type'] == 'LONG':
                        pnl = (exit_price - pos['entry_price']) * pos['quantity'] - fee
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['quantity'] - fee

                    account_balance += pnl

                    if account_balance > peak_balance:
                        peak_balance = account_balance

                    # Set cooldown
                    pair_cooldowns[symbol] = current_time + timedelta(minutes=COOLDOWN_MINUTES)

                    closed_trades.append({
                        'symbol': symbol,
                        'type': pos['type'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'balance': account_balance,
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'date': str(current_date)
                    })

                    del open_positions[symbol]
                    continue

            # Check for new entries
            if symbol in open_positions:
                continue

            if len(open_positions) >= max_concurrent_positions:
                continue

            if symbol in daily_traded_pairs.get(current_date, set()):
                continue

            # Check cooldown
            if symbol in pair_cooldowns:
                if current_time < pair_cooldowns[symbol]:
                    continue

            # Check for breakout above/below the 4-hour zone
            signal_type = None
            if close_price > bz['high']:
                signal_type = 'LONG'
            elif close_price < bz['low']:
                signal_type = 'SHORT'

            if signal_type:
                # Calculate position size
                entry_price = close_price * (1 + SLIPPAGE_PCT) if signal_type == 'LONG' else close_price * (1 - SLIPPAGE_PCT)

                if signal_type == 'LONG':
                    stop_loss = entry_price * (1 - SL_BUFFER)
                    risk_per_unit = entry_price - stop_loss
                    take_profit = entry_price + (2 * risk_per_unit)
                else:
                    stop_loss = entry_price * (1 + SL_BUFFER)
                    risk_per_unit = stop_loss - entry_price
                    take_profit = entry_price - (2 * risk_per_unit)

                # Position sizing
                risk_amount = account_balance * risk_per_trade_pct
                quantity = risk_amount / risk_per_unit

                # Cap position value
                position_value = quantity * entry_price
                if position_value > MAX_POSITION_VALUE:
                    quantity = MAX_POSITION_VALUE / entry_price

                # Entry fee
                fee = entry_price * quantity * FEE_PCT
                account_balance -= fee

                open_positions[symbol] = {
                    'type': signal_type,
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'sl': stop_loss,
                    'tp': take_profit,
                    'quantity': quantity
                }

                daily_traded_pairs[current_date].add(symbol)

    # END OF DAY: Close all remaining positions at market close
    for symbol in list(open_positions.keys()):
        pos = open_positions[symbol]

        # Get last candle of the day for this symbol
        if symbol in all_data:
            day_data = all_data[symbol][all_data[symbol].index.date == current_date]
            if len(day_data) > 0:
                last_candle = day_data.iloc[-1]
                exit_price = last_candle['Close']

                # Apply slippage
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
print("BACKTEST RESULTS - 4-HOUR BREAKING ZONE")
print("="*70)

total_trades = len(closed_trades)
winning_trades = [t for t in closed_trades if t['pnl'] > 0]
losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
total_pnl = account_balance - starting_balance
roi = (total_pnl / starting_balance) * 100

# Profit factor
gross_profit = sum(t['pnl'] for t in winning_trades)
gross_loss = abs(sum(t['pnl'] for t in losing_trades))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

# Max drawdown
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

# Exit breakdown
tp_exits = len([t for t in closed_trades if t.get('exit_reason') == 'TAKE_PROFIT'])
sl_exits = len([t for t in closed_trades if t.get('exit_reason') == 'STOP_LOSS'])
eod_exits = len([t for t in closed_trades if t.get('exit_reason') == 'END_OF_DAY'])
print(f"\nTake Profit Exits: {tp_exits}")
print(f"Stop Loss Exits:   {sl_exits}")
print(f"End of Day Exits:  {eod_exits}")

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

# Sort by P&L
sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)

for symbol, stats in sorted_symbols[:15]:
    wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
    print(f"{symbol:12} | Trades: {stats['trades']:3} | P&L: ${stats['pnl']:+8.2f} | WR: {wr:5.1f}%")

# Save results
results = {
    'config': {
        'strategy': '4-Hour Breaking Zone',
        'starting_balance': starting_balance,
        'sl_buffer': SL_BUFFER,
        'risk_per_trade': risk_per_trade_pct,
        'zone_hours': 4,
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
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'eod_exits': eod_exits
    },
    'trades': closed_trades
}

with open('backtest_yahoo_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to backtest_yahoo_results.json")
print("\n" + "="*70)
