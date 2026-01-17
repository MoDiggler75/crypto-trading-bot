#!/usr/bin/env python3
"""
BACKTEST: 4-Hour Breakout Zone Retest Strategy V2

VARIABLES:
- A = Highest CLOSING price of first 4-hour candle (top of zone)
- B = Lowest CLOSING price of first 4-hour candle (bottom of zone)
- N = The breakout zone (space between A and B)
- F = First candle that CLOSES outside the zone (the breakout signal)
- D = Candle that CLOSES back inside the zone (the entry signal)
- S = Highest/lowest CLOSING price of candles between F and D (stop loss)
- T = Take profit price = calculated from |D - S| × 2

LOGIC:
1. Day runs 00:01 to 00:00 (midnight to midnight)
2. Take first COMPLETED 4-hour candle after 00:01
3. A = close price, B = open price (or vice versa depending on candle direction)
4. Wait for F (close outside zone)
5. Wait for D (close back inside zone) - this is ENTRY
6. S = extreme closing price between F and D
7. T = D +/- (|D - S| × 2)
8. Filter out doji candles (body must be meaningful size)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pytz

print("\n" + "="*70)
print("4-HOUR BREAKOUT ZONE RETEST V2")
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

# Minimum candle body size (as percentage of price) to filter dojis
MIN_BODY_PCT = 0.001  # 0.1% minimum body size

# Cooldown after trade
COOLDOWN_MINUTES = 30

print(f"Starting Balance: ${starting_balance:,.2f}")
print(f"Max Positions: {max_concurrent_positions}")
print(f"Risk per Trade: {risk_per_trade_pct*100}%")
print(f"Min Candle Body: {MIN_BODY_PCT*100}%")
print(f"\nStrategy:")
print(f"  Zone: First 4-hour candle of the day (A=high close, B=low close)")
print(f"  F = Breakout candle (closes outside zone)")
print(f"  D = Re-entry candle (closes back inside zone) = ENTRY")
print(f"  S = Extreme closing price between F and D = STOP LOSS")
print(f"  T = D +/- (|D-S| x 2) = TAKE PROFIT")

# ============================================
# SYMBOLS TO TEST
# ============================================

# Batch 1: Major coins
CRYPTO_BATCH_1 = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD', 'LTC-USD', 'ATOM-USD']

# Batch 2: Mid-cap altcoins
CRYPTO_BATCH_2 = ['XLM-USD', 'ALGO-USD', 'AAVE-USD', 'DOGE-USD', 'SHIB-USD', 'BCH-USD', 'ETC-USD', 'FIL-USD', 'NEAR-USD', 'ICP-USD']

# Batch 3: DeFi & Layer 2
CRYPTO_BATCH_3 = ['ARB-USD', 'OP-USD', 'MKR-USD', 'SNX-USD', 'CRV-USD', 'LDO-USD', 'RUNE-USD', 'INJ-USD', 'APE-USD', 'GRT-USD']

# Batch 4: Gaming & Metaverse
CRYPTO_BATCH_4 = ['SAND-USD', 'MANA-USD', 'AXS-USD', 'IMX-USD', 'GALA-USD', 'ENJ-USD', 'RENDER-USD', 'FET-USD', 'AGIX-USD', 'OCEAN-USD']

# Batch 5: Infrastructure & Storage
CRYPTO_BATCH_5 = ['AR-USD', 'THETA-USD', 'VET-USD', 'HBAR-USD', 'QNT-USD', 'EGLD-USD', 'XTZ-USD', 'EOS-USD', 'NEO-USD', 'ZEC-USD']

# Batch 6: More altcoins
CRYPTO_BATCH_6 = ['DASH-USD', 'XMR-USD', 'COMP-USD', 'YFI-USD', 'UMA-USD', 'BAL-USD', 'SUSHI-USD', '1INCH-USD', 'ZRX-USD', 'BAT-USD']

# Select which batch to run (change this number 1-6)
BATCH_NUMBER = 1
CRYPTO_BATCHES = {
    1: CRYPTO_BATCH_1,
    2: CRYPTO_BATCH_2,
    3: CRYPTO_BATCH_3,
    4: CRYPTO_BATCH_4,
    5: CRYPTO_BATCH_5,
    6: CRYPTO_BATCH_6
}
CRYPTO_SYMBOLS = CRYPTO_BATCHES[BATCH_NUMBER]

STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD',
    'SPY', 'QQQ', 'NFLX', 'CRM', 'INTC', 'JPM', 'V', 'MA',
    'BA', 'DIS', 'COIN', 'UBER'
]

USE_CRYPTO = True
symbols_to_test = CRYPTO_SYMBOLS if USE_CRYPTO else STOCK_SYMBOLS

# For stocks: market opens 9:30, first 4h candle ends at 13:30
# For crypto: 24/7, first 4h candle is 00:00-04:00
MARKET_OPEN_HOUR = 0 if USE_CRYPTO else 9
TRADING_START_HOUR = 4 if USE_CRYPTO else 13  # After first 4h candle completes

print(f"\nTesting {len(symbols_to_test)} symbols")
print(f"Market: {'Crypto (24/7)' if USE_CRYPTO else 'Stocks (9:30-16:00)'}")

# ============================================
# HELPER FUNCTIONS
# ============================================

# Price precision - use 5 decimal places
PRICE_DECIMALS = 5

def round_price(price):
    """Round price to 5 decimal places for consistent calculations"""
    return round(price, PRICE_DECIMALS)

def format_price(price):
    """Format price with 5 decimal places for display"""
    return f"{price:.5f}"

def is_valid_candle(candle, min_body_pct=MIN_BODY_PCT):
    """Check if candle has meaningful body (not a doji)"""
    open_price = round_price(candle['Open'])
    close_price = round_price(candle['Close'])
    body_size = abs(close_price - open_price)
    body_pct = body_size / open_price if open_price > 0 else 0
    return body_pct >= min_body_pct

def get_candle_close_high_low(candle):
    """Get the high and low based on CLOSING prices (open vs close)"""
    open_price = round_price(candle['Open'])
    close_price = round_price(candle['Close'])
    return max(open_price, close_price), min(open_price, close_price)

# ============================================
# DOWNLOAD DATA - 4 HOUR CANDLES
# ============================================

print(f"\n[STEP 1] Downloading 4-hour candle data from Yahoo Finance...")

all_data_4h = {}
all_data_5m = {}
period = "60d"

for symbol in symbols_to_test:
    try:
        print(f"  Downloading {symbol}...", end=" ")
        ticker = yf.Ticker(symbol)

        # Get 4-hour candles for zone establishment
        df_4h = ticker.history(period=period, interval="1h")  # Will aggregate to 4h

        # Get 5-minute candles for trade signals
        df_5m = ticker.history(period=period, interval="5m")

        if len(df_4h) > 0 and len(df_5m) > 0:
            # Ensure timezone aware
            if df_4h.index.tz is None:
                df_4h.index = df_4h.index.tz_localize('America/New_York')
            else:
                df_4h.index = df_4h.index.tz_convert('America/New_York')

            if df_5m.index.tz is None:
                df_5m.index = df_5m.index.tz_localize('America/New_York')
            else:
                df_5m.index = df_5m.index.tz_convert('America/New_York')

            all_data_4h[symbol] = df_4h
            all_data_5m[symbol] = df_5m
            print(f"4h: {len(df_4h)}, 5m: {len(df_5m)} candles")
        else:
            print("No data")
    except Exception as e:
        print(f"Error: {e}")

print(f"\nLoaded data for {len(all_data_4h)} symbols")

if len(all_data_4h) == 0:
    print("ERROR: No data loaded. Exiting.")
    exit(1)

# ============================================
# AGGREGATE 1H TO 4H CANDLES
# ============================================

print(f"\n[STEP 2] Aggregating hourly data to 4-hour candles...")

all_data_4h_agg = {}

for symbol, df_1h in all_data_4h.items():
    # Resample to 4-hour candles starting at midnight
    df_4h = df_1h.resample('4h', origin='start_day').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    all_data_4h_agg[symbol] = df_4h
    print(f"  {symbol}: {len(df_4h)} 4-hour candles")

# ============================================
# BACKTEST ENGINE
# ============================================

print(f"\n[STEP 3] Running backtest...")

# State
account_balance = starting_balance
peak_balance = starting_balance
open_positions = {}
closed_trades = []
pair_cooldowns = {}

# Tracking state per symbol
# States: 'waiting_for_zone', 'waiting_for_F', 'waiting_for_D'
symbol_states = {}

# Get all unique dates
all_dates = set()
for symbol, df in all_data_5m.items():
    all_dates.update(df.index.date)

all_dates = sorted(all_dates)
print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
print(f"Total trading days: {len(all_dates)}")

# Stats
stats = {
    'zones_established': 0,
    'F_signals': 0,
    'D_signals': 0,
    'long_trades': 0,
    'short_trades': 0,
    'doji_filtered': 0
}

# Process each day
for current_date in all_dates:

    # ============================================
    # ESTABLISH ZONE FROM FIRST 4-HOUR CANDLE
    # ============================================

    for symbol in all_data_4h_agg.keys():
        df_4h = all_data_4h_agg[symbol]

        # Get 4-hour candles for this day
        day_4h = df_4h[df_4h.index.date == current_date]

        if len(day_4h) == 0:
            continue

        # Get the FIRST completed 4-hour candle of the day
        # For crypto, this would be the 00:00-04:00 candle
        # For stocks, this would be the 09:30-13:30 candle
        # Filter to only candles starting at/after market open hour
        day_4h_filtered = day_4h[day_4h.index.hour >= MARKET_OPEN_HOUR]
        if len(day_4h_filtered) == 0:
            continue
        first_4h_candle = day_4h_filtered.iloc[0]

        # Check if it's a valid candle (not a doji)
        if not is_valid_candle(first_4h_candle):
            stats['doji_filtered'] += 1
            continue

        # A = highest closing price (max of open, close)
        # B = lowest closing price (min of open, close)
        A, B = get_candle_close_high_low(first_4h_candle)

        # Initialize state for this symbol/day
        symbol_states[symbol] = {
            'A': A,  # Top of zone (highest close)
            'B': B,  # Bottom of zone (lowest close)
            'N': A - B,  # Zone size
            'zone_date': current_date,
            'state': 'waiting_for_F',
            'F_candle': None,  # Breakout candle
            'F_direction': None,  # 'above' or 'below'
            'sequence_candles': [],  # Candles between F and D
            'sequence_closes': [],  # Closing prices between F and D
        }
        stats['zones_established'] += 1

    # ============================================
    # PROCESS 5-MINUTE CANDLES FOR SIGNALS
    # ============================================

    for symbol in all_data_5m.keys():
        if symbol not in symbol_states:
            continue

        ss = symbol_states[symbol]

        # Skip if zone is from a different day
        if ss['zone_date'] != current_date:
            continue

        df_5m = all_data_5m[symbol]

        # Get 5-minute candles for today (after the first 4-hour candle closes)
        # Crypto: First 4h candle is 00:00-04:00, so start at 04:00
        # Stocks: First 4h candle is 09:30-13:30, so start at 13:30
        day_5m = df_5m[
            (df_5m.index.date == current_date) &
            (df_5m.index.hour >= TRADING_START_HOUR)
        ]

        A = ss['A']
        B = ss['B']

        for idx, candle in day_5m.iterrows():
            current_time = idx
            close_price = round_price(candle['Close'])
            open_price = round_price(candle['Open'])
            high_price = round_price(candle['High'])
            low_price = round_price(candle['Low'])

            # Skip doji candles
            if not is_valid_candle(candle):
                continue

            # ============================================
            # CHECK EXITS FOR OPEN POSITIONS
            # ============================================

            if symbol in open_positions:
                pos = open_positions[symbol]
                exit_price = None
                exit_reason = None

                # S is the stop loss price
                # T is the take profit price
                S = pos['S']
                T = pos['T']

                if pos['type'] == 'LONG':
                    # Stop loss triggered when price TOUCHES S (goes below)
                    if low_price <= S:
                        exit_price = S
                        exit_reason = 'STOP_LOSS'
                    # Take profit triggered when price TOUCHES T (goes above)
                    elif high_price >= T:
                        exit_price = T
                        exit_reason = 'TAKE_PROFIT'

                else:  # SHORT
                    # Stop loss triggered when price TOUCHES S (goes above)
                    if high_price >= S:
                        exit_price = S
                        exit_reason = 'STOP_LOSS'
                    # Take profit triggered when price TOUCHES T (goes below)
                    elif low_price <= T:
                        exit_price = T
                        exit_reason = 'TAKE_PROFIT'

                if exit_price:
                    # Apply slippage
                    if pos['type'] == 'LONG':
                        exit_price = round_price(exit_price * (1 - SLIPPAGE_PCT))
                    else:
                        exit_price = round_price(exit_price * (1 + SLIPPAGE_PCT))

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
                        'entry_price': pos['D'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'A': pos['A'],
                        'B': pos['B'],
                        'S': S,
                        'T': T,
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'balance': account_balance,
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'date': str(current_date)
                    })

                    del open_positions[symbol]
                    # Reset state
                    ss['state'] = 'waiting_for_F'
                    ss['F_candle'] = None
                    ss['F_direction'] = None
                    ss['sequence_candles'] = []
                    ss['sequence_closes'] = []
                    continue

            # ============================================
            # STATE MACHINE: WAITING FOR F (BREAKOUT)
            # ============================================

            if ss['state'] == 'waiting_for_F':
                # Check if candle CLOSES outside the zone
                if close_price > A:
                    # F = Breakout ABOVE the zone
                    ss['state'] = 'waiting_for_D'
                    ss['F_candle'] = candle
                    ss['F_direction'] = 'above'
                    ss['sequence_candles'] = [candle]
                    ss['sequence_closes'] = [close_price]
                    stats['F_signals'] += 1

                elif close_price < B:
                    # F = Breakout BELOW the zone
                    ss['state'] = 'waiting_for_D'
                    ss['F_candle'] = candle
                    ss['F_direction'] = 'below'
                    ss['sequence_candles'] = [candle]
                    ss['sequence_closes'] = [close_price]
                    stats['F_signals'] += 1

            # ============================================
            # STATE MACHINE: WAITING FOR D (RE-ENTRY)
            # ============================================

            elif ss['state'] == 'waiting_for_D':
                # Add candle to sequence
                ss['sequence_candles'].append(candle)
                ss['sequence_closes'].append(close_price)

                # Check if candle CLOSES back inside the zone
                if B <= close_price <= A:
                    # D = Re-entry candle - THIS IS THE ENTRY SIGNAL
                    stats['D_signals'] += 1

                    D = round_price(close_price)  # Entry price (close of D candle)

                    # S = extreme closing price between F and D
                    if ss['F_direction'] == 'above':
                        # Breakout was above, so S = highest close in sequence
                        S = round_price(max(ss['sequence_closes']))
                        # This is a SHORT trade (price went up, now coming down)
                        trade_type = 'SHORT'
                        # T = D - (|D - S| × 2)
                        distance = round_price(S - D)
                        T = round_price(D - (distance * 2))
                    else:
                        # Breakout was below, so S = lowest close in sequence
                        S = round_price(min(ss['sequence_closes']))
                        # This is a LONG trade (price went down, now coming up)
                        trade_type = 'LONG'
                        # T = D + (|D - S| × 2)
                        distance = round_price(D - S)
                        T = round_price(D + (distance * 2))

                    # Check if we can take the trade
                    can_trade = True
                    if symbol in open_positions:
                        can_trade = False
                    if len(open_positions) >= max_concurrent_positions:
                        can_trade = False
                    if symbol in pair_cooldowns and current_time < pair_cooldowns[symbol]:
                        can_trade = False
                    if distance <= 0:
                        can_trade = False  # Invalid S calculation

                    if can_trade:
                        # Position sizing
                        risk_per_unit = distance
                        risk_amount = account_balance * risk_per_trade_pct
                        quantity = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                        # Cap position value
                        position_value = quantity * D
                        if position_value > MAX_POSITION_VALUE:
                            quantity = MAX_POSITION_VALUE / D

                        if quantity > 0:
                            # Apply slippage to entry
                            entry_price = round_price(D * (1 + SLIPPAGE_PCT) if trade_type == 'LONG' else D * (1 - SLIPPAGE_PCT))

                            # Entry fee
                            fee = entry_price * quantity * FEE_PCT
                            account_balance -= fee

                            if trade_type == 'LONG':
                                stats['long_trades'] += 1
                            else:
                                stats['short_trades'] += 1

                            open_positions[symbol] = {
                                'type': trade_type,
                                'entry_price': entry_price,
                                'D': D,
                                'S': S,
                                'T': T,
                                'A': A,
                                'B': B,
                                'entry_time': current_time,
                                'quantity': quantity,
                                'distance': distance
                            }

                    # Reset state (whether we traded or not)
                    ss['state'] = 'waiting_for_F'
                    ss['F_candle'] = None
                    ss['F_direction'] = None
                    ss['sequence_candles'] = []
                    ss['sequence_closes'] = []

# ============================================
# CLOSE REMAINING POSITIONS
# ============================================

print(f"\nClosing {len(open_positions)} remaining positions...")

for symbol in list(open_positions.keys()):
    pos = open_positions[symbol]

    if symbol in all_data_5m:
        last_candle = all_data_5m[symbol].iloc[-1]
        exit_price = round_price(last_candle['Close'])

        if pos['type'] == 'LONG':
            exit_price = round_price(exit_price * (1 - SLIPPAGE_PCT))
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
        else:
            exit_price = round_price(exit_price * (1 + SLIPPAGE_PCT))
            pnl = (pos['entry_price'] - exit_price) * pos['quantity']

        fee = exit_price * pos['quantity'] * FEE_PCT
        pnl -= fee
        account_balance += pnl

        closed_trades.append({
            'symbol': symbol,
            'type': pos['type'],
            'entry_price': pos['D'],
            'exit_price': exit_price,
            'exit_reason': 'END_OF_BACKTEST',
            'A': pos['A'],
            'B': pos['B'],
            'S': pos['S'],
            'T': pos['T'],
            'quantity': pos['quantity'],
            'pnl': pnl,
            'balance': account_balance
        })

    del open_positions[symbol]

# ============================================
# RESULTS
# ============================================

print("\n" + "="*70)
print("BACKTEST RESULTS - 4-HOUR BREAKOUT ZONE V2")
print("="*70)

total_trades = len(closed_trades)

if total_trades == 0:
    print("\nNo trades executed!")
    print(f"\nStats:")
    print(f"  Zones established: {stats['zones_established']}")
    print(f"  F signals (breakouts): {stats['F_signals']}")
    print(f"  D signals (re-entries): {stats['D_signals']}")
    print(f"  Doji candles filtered: {stats['doji_filtered']}")
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

# Stats
print(f"\n--- SIGNAL ANALYSIS ---")
print(f"Zones Established: {stats['zones_established']}")
print(f"F Signals (breakouts): {stats['F_signals']}")
print(f"D Signals (re-entries): {stats['D_signals']}")
print(f"LONG Trades: {stats['long_trades']}")
print(f"SHORT Trades: {stats['short_trades']}")
print(f"Doji Candles Filtered: {stats['doji_filtered']}")

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
    print(f"  Zone: A=${trade['A']:.5f}, B=${trade['B']:.5f}")
    print(f"  Entry (D): ${trade['entry_price']:.5f}")
    print(f"  Stop Loss (S): ${trade['S']:.5f}")
    print(f"  Take Profit (T): ${trade['T']:.5f}")
    print(f"  Exit: ${trade['exit_price']:.5f} ({trade['exit_reason']})")
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

for symbol, s in sorted_symbols:
    wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
    print(f"{symbol:12} | Trades: {s['trades']:3} | P&L: ${s['pnl']:+8.2f} | WR: {wr:5.1f}%")

# Save results
results = {
    'config': {
        'strategy': '4-Hour Breakout Zone V2',
        'variables': 'A=zone high, B=zone low, F=breakout, D=entry, S=stop loss, T=take profit',
        'starting_balance': starting_balance,
        'risk_per_trade': risk_per_trade_pct,
        'min_body_pct': MIN_BODY_PCT,
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

with open('backtest_4hr_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to backtest_4hr_v2_results.json")
print("\n" + "="*70)
