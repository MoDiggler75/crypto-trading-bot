#!/usr/bin/env python3
"""
BACKTEST: Breakout Zone Retest Strategy
Using Yahoo Finance Data

Strategy:
1. First 4 hours (9:30 AM - 1:30 PM) establishes the breakout zone (high/low)
2. Wait for price to CLOSE OUTSIDE the zone (first breakout)
3. Track all candles in the "sequence" after breakout
4. Wait for price to CLOSE BACK INSIDE the zone (re-entry = TRADE SIGNAL)
5. Direction:
   - Breakout at TOP (above zone high) → SHORT
   - Breakout at BOTTOM (below zone low) → LONG
6. Stop Loss = Extreme price of the sequence
   - SHORT: Highest price in sequence
   - LONG: Lowest price in sequence
7. Take Profit = Entry ± (2 × distance to stop loss)
8. Carry trades overnight (no forced EOD close)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pytz

print("\n" + "="*70)
print("BREAKOUT ZONE RETEST STRATEGY - YAHOO FINANCE")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

starting_balance = 10000.0
max_concurrent_positions = 5
risk_per_trade_pct = 0.05

# Safety limits
MAX_POSITION_VALUE = starting_balance * 0.30

# Fees
FEE_PCT = 0.001
SLIPPAGE_PCT = 0.0005

# Cooldown
COOLDOWN_MINUTES = 30

# Market Times (adjusted for crypto - 9:00 AM ET start)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_HOUR = 16

# 4-Hour Zone: 9:00 AM to 1:00 PM ET
ZONE_END_HOUR = 13
ZONE_END_MINUTE = 0

print(f"Starting Balance: ${starting_balance:,.2f}")
print(f"Max Positions: {max_concurrent_positions}")
print(f"Risk per Trade: {risk_per_trade_pct*100}%")
print(f"\nStrategy:")
print(f"  4-Hour Zone: 9:00 AM - 1:00 PM ET (establishes high/low)")
print(f"  Signal: Breakout -> Re-entry back into zone")
print(f"  TOP breakout + re-entry = SHORT")
print(f"  BOTTOM breakout + re-entry = LONG")
print(f"  Stop Loss: Extreme of breakout sequence")
print(f"  Take Profit: 2x the stop loss distance")
print(f"  Trades carry overnight (no forced close)")

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

USE_CRYPTO = True  # Changed to True for crypto
symbols_to_test = CRYPTO_SYMBOLS if USE_CRYPTO else STOCK_SYMBOLS

print(f"\nTesting {len(symbols_to_test)} symbols")

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
open_positions = {}  # {symbol: position_data}
closed_trades = []
pair_cooldowns = {}  # {symbol: datetime}

# Breakout tracking per symbol
# States: 'waiting_for_breakout', 'tracking_top_sequence', 'tracking_bottom_sequence'
breakout_states = {}

# Structure for each symbol's breakout tracking:
# {
#     'state': 'waiting_for_breakout' | 'tracking_top_sequence' | 'tracking_bottom_sequence',
#     'zone_high': float,
#     'zone_low': float,
#     'sequence_candles': [],  # All candles in the sequence
#     'sequence_high': float,  # Highest price in sequence
#     'sequence_low': float,   # Lowest price in sequence
# }

# Get all unique dates
all_dates = set()
for symbol, df in all_data.items():
    all_dates.update(df.index.date)

all_dates = sorted(all_dates)
print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
print(f"Total trading days: {len(all_dates)}")

# Counters
stats = {
    'zones_established': 0,
    'top_breakouts': 0,
    'bottom_breakouts': 0,
    'top_reentries': 0,
    'bottom_reentries': 0,
    'long_signals': 0,
    'short_signals': 0
}

# Process each day
for current_date in all_dates:

    # ============================================
    # BUILD 4-HOUR BREAKING ZONES FOR NEW DAY
    # ============================================

    for symbol, df in all_data.items():
        day_data = df[df.index.date == current_date]

        if len(day_data) == 0:
            continue

        # Get all candles in the 4-hour zone (9:30 AM to 1:30 PM)
        zone_candles = day_data[
            (day_data.index.hour >= MARKET_OPEN_HOUR) &
            (
                (day_data.index.hour < ZONE_END_HOUR) |
                ((day_data.index.hour == ZONE_END_HOUR) & (day_data.index.minute < ZONE_END_MINUTE))
            )
        ]

        if len(zone_candles) < 10:
            continue

        # Establish new zone for this symbol
        zone_high = zone_candles['High'].max()
        zone_low = zone_candles['Low'].min()

        breakout_states[symbol] = {
            'state': 'waiting_for_breakout',
            'zone_high': zone_high,
            'zone_low': zone_low,
            'zone_date': current_date,
            'sequence_candles': [],
            'sequence_high': None,
            'sequence_low': None
        }
        stats['zones_established'] += 1

    # ============================================
    # PROCESS ALL CANDLES FOR THE DAY
    # ============================================

    for symbol, df in all_data.items():
        if symbol not in breakout_states:
            continue

        bs = breakout_states[symbol]
        day_data = df[df.index.date == current_date]

        # Process candles after zone closes (1:30 PM onwards) OR all day if zone was from previous day
        if bs['zone_date'] == current_date:
            # New zone today - only process after zone closes
            zone_close_time = day_data[
                (day_data.index.hour == ZONE_END_HOUR) &
                (day_data.index.minute >= ZONE_END_MINUTE)
            ]
            if len(zone_close_time) > 0:
                trading_start = zone_close_time.index[0]
            else:
                continue
            trading_candles = day_data[day_data.index >= trading_start]
        else:
            # Zone from previous day - process all candles
            trading_candles = day_data

        for idx, candle in trading_candles.iterrows():
            current_time = idx
            close_price = candle['Close']
            high_price = candle['High']
            low_price = candle['Low']

            zone_high = bs['zone_high']
            zone_low = bs['zone_low']

            # ============================================
            # CHECK EXITS FOR OPEN POSITIONS
            # ============================================
            if symbol in open_positions:
                pos = open_positions[symbol]
                exit_price = None
                exit_reason = None

                if pos['type'] == 'LONG':
                    # Check stop loss (price goes below SL)
                    if low_price <= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'STOP_LOSS'
                    # Check take profit (price goes above TP)
                    elif high_price >= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'TAKE_PROFIT'

                else:  # SHORT
                    # Check stop loss (price goes above SL)
                    if high_price >= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'STOP_LOSS'
                    # Check take profit (price goes below TP)
                    elif low_price <= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'TAKE_PROFIT'

                if exit_price:
                    # Apply slippage
                    if pos['type'] == 'LONG':
                        exit_price = exit_price * (1 - SLIPPAGE_PCT)
                    else:
                        exit_price = exit_price * (1 + SLIPPAGE_PCT)

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
                        'sl': pos['sl'],
                        'tp': pos['tp'],
                        'zone_high': pos['zone_high'],
                        'zone_low': pos['zone_low'],
                        'sequence_extreme': pos['sequence_extreme'],
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'balance': account_balance,
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'entry_date': str(pos.get('entry_date', '')),
                        'exit_date': str(current_date)
                    })

                    del open_positions[symbol]
                    # Reset breakout state after trade closes
                    bs['state'] = 'waiting_for_breakout'
                    bs['sequence_candles'] = []
                    bs['sequence_high'] = None
                    bs['sequence_low'] = None
                    continue

            # ============================================
            # BREAKOUT STATE MACHINE
            # ============================================

            current_state = bs['state']

            # Determine if price is inside or outside zone
            price_above_zone = close_price > zone_high
            price_below_zone = close_price < zone_low
            price_inside_zone = zone_low <= close_price <= zone_high

            if current_state == 'waiting_for_breakout':
                # Looking for first breakout
                if price_above_zone:
                    # TOP BREAKOUT - start tracking sequence
                    bs['state'] = 'tracking_top_sequence'
                    bs['sequence_candles'] = [candle]
                    bs['sequence_high'] = high_price
                    bs['sequence_low'] = low_price
                    stats['top_breakouts'] += 1

                elif price_below_zone:
                    # BOTTOM BREAKOUT - start tracking sequence
                    bs['state'] = 'tracking_bottom_sequence'
                    bs['sequence_candles'] = [candle]
                    bs['sequence_high'] = high_price
                    bs['sequence_low'] = low_price
                    stats['bottom_breakouts'] += 1

            elif current_state == 'tracking_top_sequence':
                # Tracking after TOP breakout, waiting for re-entry
                bs['sequence_candles'].append(candle)
                bs['sequence_high'] = max(bs['sequence_high'], high_price)
                bs['sequence_low'] = min(bs['sequence_low'], low_price)

                if price_inside_zone:
                    # RE-ENTRY! Price closed back inside zone
                    # This is a SHORT signal (breakout was at top)
                    stats['top_reentries'] += 1

                    # Check if we can take the trade
                    can_trade = True
                    if symbol in open_positions:
                        can_trade = False
                    if len(open_positions) >= max_concurrent_positions:
                        can_trade = False
                    if symbol in pair_cooldowns and current_time < pair_cooldowns[symbol]:
                        can_trade = False

                    if can_trade:
                        stats['short_signals'] += 1

                        # Entry price = close of re-entry candle (with slippage for short)
                        entry_price = close_price * (1 - SLIPPAGE_PCT)

                        # Stop loss = highest price in the sequence
                        stop_loss = bs['sequence_high']

                        # Distance from entry to stop loss
                        sl_distance = stop_loss - entry_price

                        # Take profit = entry - (2 × distance)
                        take_profit = entry_price - (2 * sl_distance)

                        # Position sizing based on risk
                        risk_per_unit = sl_distance
                        risk_amount = account_balance * risk_per_trade_pct
                        quantity = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                        # Cap position value
                        position_value = quantity * entry_price
                        if position_value > MAX_POSITION_VALUE:
                            quantity = MAX_POSITION_VALUE / entry_price

                        if quantity > 0:
                            # Entry fee
                            fee = entry_price * quantity * FEE_PCT
                            account_balance -= fee

                            open_positions[symbol] = {
                                'type': 'SHORT',
                                'entry_price': entry_price,
                                'entry_time': current_time,
                                'entry_date': current_date,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'quantity': quantity,
                                'zone_high': zone_high,
                                'zone_low': zone_low,
                                'sequence_extreme': stop_loss,
                                'sl_distance': sl_distance
                            }

                    # Reset state after re-entry (whether we traded or not)
                    bs['state'] = 'waiting_for_breakout'
                    bs['sequence_candles'] = []
                    bs['sequence_high'] = None
                    bs['sequence_low'] = None

            elif current_state == 'tracking_bottom_sequence':
                # Tracking after BOTTOM breakout, waiting for re-entry
                bs['sequence_candles'].append(candle)
                bs['sequence_high'] = max(bs['sequence_high'], high_price)
                bs['sequence_low'] = min(bs['sequence_low'], low_price)

                if price_inside_zone:
                    # RE-ENTRY! Price closed back inside zone
                    # This is a LONG signal (breakout was at bottom)
                    stats['bottom_reentries'] += 1

                    # Check if we can take the trade
                    can_trade = True
                    if symbol in open_positions:
                        can_trade = False
                    if len(open_positions) >= max_concurrent_positions:
                        can_trade = False
                    if symbol in pair_cooldowns and current_time < pair_cooldowns[symbol]:
                        can_trade = False

                    if can_trade:
                        stats['long_signals'] += 1

                        # Entry price = close of re-entry candle (with slippage for long)
                        entry_price = close_price * (1 + SLIPPAGE_PCT)

                        # Stop loss = lowest price in the sequence
                        stop_loss = bs['sequence_low']

                        # Distance from entry to stop loss
                        sl_distance = entry_price - stop_loss

                        # Take profit = entry + (2 × distance)
                        take_profit = entry_price + (2 * sl_distance)

                        # Position sizing based on risk
                        risk_per_unit = sl_distance
                        risk_amount = account_balance * risk_per_trade_pct
                        quantity = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                        # Cap position value
                        position_value = quantity * entry_price
                        if position_value > MAX_POSITION_VALUE:
                            quantity = MAX_POSITION_VALUE / entry_price

                        if quantity > 0:
                            # Entry fee
                            fee = entry_price * quantity * FEE_PCT
                            account_balance -= fee

                            open_positions[symbol] = {
                                'type': 'LONG',
                                'entry_price': entry_price,
                                'entry_time': current_time,
                                'entry_date': current_date,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'quantity': quantity,
                                'zone_high': zone_high,
                                'zone_low': zone_low,
                                'sequence_extreme': stop_loss,
                                'sl_distance': sl_distance
                            }

                    # Reset state after re-entry
                    bs['state'] = 'waiting_for_breakout'
                    bs['sequence_candles'] = []
                    bs['sequence_high'] = None
                    bs['sequence_low'] = None

# ============================================
# CLOSE ANY REMAINING POSITIONS AT END OF BACKTEST
# ============================================
print(f"\nClosing {len(open_positions)} remaining positions at end of backtest...")

for symbol in list(open_positions.keys()):
    pos = open_positions[symbol]

    if symbol in all_data:
        last_candle = all_data[symbol].iloc[-1]
        exit_price = last_candle['Close']

        if pos['type'] == 'LONG':
            exit_price = exit_price * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
        else:
            exit_price = exit_price * (1 + SLIPPAGE_PCT)
            pnl = (pos['entry_price'] - exit_price) * pos['quantity']

        fee = exit_price * pos['quantity'] * FEE_PCT
        pnl -= fee
        account_balance += pnl

        closed_trades.append({
            'symbol': symbol,
            'type': pos['type'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'exit_reason': 'END_OF_BACKTEST',
            'sl': pos['sl'],
            'tp': pos['tp'],
            'zone_high': pos['zone_high'],
            'zone_low': pos['zone_low'],
            'sequence_extreme': pos['sequence_extreme'],
            'quantity': pos['quantity'],
            'pnl': pnl,
            'balance': account_balance,
            'entry_time': pos['entry_time'],
            'exit_time': last_candle.name,
            'entry_date': str(pos.get('entry_date', '')),
            'exit_date': str(all_dates[-1])
        })

    del open_positions[symbol]

# ============================================
# RESULTS
# ============================================

print("\n" + "="*70)
print("BACKTEST RESULTS - BREAKOUT ZONE RETEST STRATEGY")
print("="*70)

total_trades = len(closed_trades)

if total_trades == 0:
    print("\nNo trades executed!")
    print(f"\nAnalysis:")
    print(f"  Zones established: {stats['zones_established']}")
    print(f"  Top breakouts: {stats['top_breakouts']}")
    print(f"  Bottom breakouts: {stats['bottom_breakouts']}")
    print(f"  Top re-entries: {stats['top_reentries']}")
    print(f"  Bottom re-entries: {stats['bottom_reentries']}")
    exit(0)

winning_trades = [t for t in closed_trades if t['pnl'] > 0]
losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

win_rate = len(winning_trades) / total_trades * 100
total_pnl = account_balance - starting_balance
roi = (total_pnl / starting_balance) * 100

gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
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

# Signal analysis
print(f"\n--- SIGNAL ANALYSIS ---")
print(f"Zones Established: {stats['zones_established']}")
print(f"Top Breakouts:     {stats['top_breakouts']}")
print(f"Bottom Breakouts:  {stats['bottom_breakouts']}")
print(f"Top Re-entries:    {stats['top_reentries']} (-> SHORT signals)")
print(f"Bottom Re-entries: {stats['bottom_reentries']} (-> LONG signals)")
print(f"SHORT Trades Taken: {stats['short_signals']}")
print(f"LONG Trades Taken:  {stats['long_signals']}")

# Exit breakdown
tp_exits = len([t for t in closed_trades if t.get('exit_reason') == 'TAKE_PROFIT'])
sl_exits = len([t for t in closed_trades if t.get('exit_reason') == 'STOP_LOSS'])
eob_exits = len([t for t in closed_trades if t.get('exit_reason') == 'END_OF_BACKTEST'])
print(f"\nTake Profit Exits: {tp_exits}")
print(f"Stop Loss Exits:   {sl_exits}")
print(f"End of Backtest:   {eob_exits}")

# Performance by direction
long_trades = [t for t in closed_trades if t['type'] == 'LONG']
short_trades = [t for t in closed_trades if t['type'] == 'SHORT']

if long_trades:
    long_pnl = sum(t['pnl'] for t in long_trades)
    long_wins = len([t for t in long_trades if t['pnl'] > 0])
    long_wr = long_wins / len(long_trades) * 100
    print(f"\nLONG Performance:  ${long_pnl:+.2f} ({len(long_trades)} trades, {long_wr:.1f}% WR)")

if short_trades:
    short_pnl = sum(t['pnl'] for t in short_trades)
    short_wins = len([t for t in short_trades if t['pnl'] > 0])
    short_wr = short_wins / len(short_trades) * 100
    print(f"SHORT Performance: ${short_pnl:+.2f} ({len(short_trades)} trades, {short_wr:.1f}% WR)")

# Sample trades
print("\n--- SAMPLE TRADES ---")
for trade in closed_trades[:5]:
    print(f"\n{trade['symbol']} {trade['type']}")
    print(f"  Entry: ${trade['entry_price']:.2f} @ {trade['entry_time']}")
    print(f"  Exit:  ${trade['exit_price']:.2f} ({trade['exit_reason']})")
    print(f"  Zone:  ${trade['zone_low']:.2f} - ${trade['zone_high']:.2f}")
    print(f"  SL: ${trade['sl']:.2f}, TP: ${trade['tp']:.2f}")
    print(f"  P&L: ${trade['pnl']:+.2f}")

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

for symbol, s in sorted_symbols[:15]:
    wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
    print(f"{symbol:12} | Trades: {s['trades']:3} | P&L: ${s['pnl']:+8.2f} | WR: {wr:5.1f}%")

# Save results
results = {
    'config': {
        'strategy': 'Breakout Zone Retest',
        'starting_balance': starting_balance,
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
        'stats': stats
    },
    'trades': closed_trades
}

with open('backtest_breakout_retest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to backtest_breakout_retest_results.json")
print("\n" + "="*70)
