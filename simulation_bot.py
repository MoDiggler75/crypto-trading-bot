#!/usr/bin/env python3
"""
SIMULATION BOT - 4-Hour Breakout Zone Retest Strategy
Uses live Kraken data but only paper trades (no real money)

This bot:
1. Connects to Kraken API for live price data
2. Establishes 4-hour breakout zones
3. Detects F (breakout) and D (re-entry) signals
4. Simulates trades with paper money
5. Logs all activity to file
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("\n" + "="*70)
print("SIMULATION BOT - 4-Hour Breakout Zone Retest")
print("MODE: PAPER TRADING (No real money)")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

# Paper trading account
STARTING_BALANCE = 10000.0
account_balance = STARTING_BALANCE
peak_balance = STARTING_BALANCE

# Trading settings
MAX_CONCURRENT_POSITIONS = 3
RISK_PER_TRADE_PCT = 0.05  # 5% risk per trade
MAX_POSITION_VALUE = STARTING_BALANCE * 0.30

# Fees (Kraken fees)
FEE_PCT = 0.0026  # 0.26% taker fee
SLIPPAGE_PCT = 0.0005

# Fixed Stop Loss and Take Profit percentages
STOP_LOSS_PCT = 0.02   # 2% stop loss
TAKE_PROFIT_PCT = 0.02  # 2% take profit

# Price precision
PRICE_DECIMALS = 5

# Cooldown after trade (minutes)
COOLDOWN_MINUTES = 30

# Minimum candle body size to filter dojis
MIN_BODY_PCT = 0.001

# Pairs to trade (Kraken format)
TRADING_PAIRS = [
    'XXBTZUSD',   # BTC/USD
    'XETHZUSD',   # ETH/USD
    'SOLUSD',     # SOL/USD
    'XXRPZUSD',   # XRP/USD
    'ADAUSD',     # ADA/USD
    'LINKUSD',    # LINK/USD
    'DOTUSD',     # DOT/USD
    'LTCUSD',     # LTC/USD
]

# Update interval (seconds)
UPDATE_INTERVAL = 60  # Check every minute

# Log file
LOG_FILE = 'simulation_log.json'

print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
print(f"Max Positions: {MAX_CONCURRENT_POSITIONS}")
print(f"Risk per Trade: {RISK_PER_TRADE_PCT*100}%")
print(f"Trading Pairs: {len(TRADING_PAIRS)}")
print(f"Update Interval: {UPDATE_INTERVAL}s")

# ============================================
# HELPER FUNCTIONS
# ============================================

def round_price(price):
    """Round price to 5 decimal places"""
    return round(float(price), PRICE_DECIMALS)

def is_valid_candle(open_price, close_price, min_body_pct=MIN_BODY_PCT):
    """Check if candle has meaningful body (not a doji)"""
    body_size = abs(close_price - open_price)
    body_pct = body_size / open_price if open_price > 0 else 0
    return body_pct >= min_body_pct

def get_close_high_low(open_price, close_price):
    """Get high and low based on closing prices"""
    return max(open_price, close_price), min(open_price, close_price)

def log_event(event_type, data):
    """Log events to file"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': event_type,
        'data': data
    }

    # Append to log file
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except:
            logs = []

    logs.append(log_entry)

    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2, default=str)

    return log_entry

# ============================================
# KRAKEN API SETUP
# ============================================

print("\n[SETUP] Connecting to Kraken API...")

try:
    api = krakenex.API()
    api.key = os.getenv('KRAKEN_PUBLIC_KEY')
    api.secret = os.getenv('KRAKEN_PRIVATE_KEY')
    kraken = KrakenAPI(api)
    print("[OK] Connected to Kraken API")
except Exception as e:
    print(f"[ERROR] Failed to connect: {e}")
    exit(1)

# ============================================
# STATE MANAGEMENT
# ============================================

# Track state per pair
pair_states = {}
open_positions = {}
closed_trades = []
pair_cooldowns = {}

# Stats
stats = {
    'zones_established': 0,
    'F_signals': 0,
    'D_signals': 0,
    'long_trades': 0,
    'short_trades': 0,
    'total_pnl': 0.0
}

# ============================================
# MAIN FUNCTIONS
# ============================================

def get_4h_candle(pair):
    """Get the current 4-hour candle data"""
    try:
        # Get OHLC data (4 hour = 240 minutes)
        ohlc, last = kraken.get_ohlc_data(pair, interval=240, ascending=True)

        if len(ohlc) > 0:
            # Get the last completed 4h candle
            last_candle = ohlc.iloc[-2] if len(ohlc) > 1 else ohlc.iloc[-1]
            return {
                'open': round_price(last_candle['open']),
                'high': round_price(last_candle['high']),
                'low': round_price(last_candle['low']),
                'close': round_price(last_candle['close']),
                'time': last_candle.name
            }
    except Exception as e:
        print(f"  [ERROR] {pair}: {e}")
    return None

def get_current_price(pair):
    """Get current ticker price"""
    try:
        ticker = kraken.get_ticker_information(pair)
        if pair in ticker.index:
            # 'c' is the last trade closed [price, lot volume]
            price = round_price(ticker.loc[pair, 'c'][0])
            return price
    except Exception as e:
        print(f"  [ERROR] {pair} ticker: {e}")
    return None

def establish_zone(pair):
    """Establish the 4-hour breakout zone for a pair"""
    candle = get_4h_candle(pair)

    if candle is None:
        return None

    # Check if valid candle (not doji)
    if not is_valid_candle(candle['open'], candle['close']):
        return None

    # A = highest close, B = lowest close
    A, B = get_close_high_low(candle['open'], candle['close'])

    zone = {
        'A': A,  # Top of zone
        'B': B,  # Bottom of zone
        'N': A - B,  # Zone size
        'established_at': datetime.now(),
        'candle_time': candle['time'],
        'state': 'waiting_for_F',
        'F_direction': None,
        'sequence_closes': []
    }

    stats['zones_established'] += 1
    log_event('ZONE_ESTABLISHED', {'pair': pair, 'A': A, 'B': B, 'N': A-B})

    return zone

def check_signals(pair, current_price):
    """Check for F and D signals"""
    global account_balance, peak_balance

    if pair not in pair_states:
        return

    ps = pair_states[pair]
    A = ps['A']
    B = ps['B']

    # ============================================
    # CHECK EXITS FOR OPEN POSITIONS
    # ============================================

    if pair in open_positions:
        pos = open_positions[pair]
        S = pos['S']
        T = pos['T']
        exit_price = None
        exit_reason = None

        if pos['type'] == 'LONG':
            if current_price <= S:
                exit_price = round_price(S * (1 - SLIPPAGE_PCT))
                exit_reason = 'STOP_LOSS'
            elif current_price >= T:
                exit_price = round_price(T * (1 - SLIPPAGE_PCT))
                exit_reason = 'TAKE_PROFIT'
        else:  # SHORT
            if current_price >= S:
                exit_price = round_price(S * (1 + SLIPPAGE_PCT))
                exit_reason = 'STOP_LOSS'
            elif current_price <= T:
                exit_price = round_price(T * (1 + SLIPPAGE_PCT))
                exit_reason = 'TAKE_PROFIT'

        if exit_price:
            # Calculate P&L
            fee = exit_price * pos['quantity'] * FEE_PCT

            if pos['type'] == 'LONG':
                pnl = (exit_price - pos['entry_price']) * pos['quantity'] - fee
            else:
                pnl = (pos['entry_price'] - exit_price) * pos['quantity'] - fee

            account_balance += pnl
            if account_balance > peak_balance:
                peak_balance = account_balance

            stats['total_pnl'] += pnl

            trade_result = {
                'pair': pair,
                'type': pos['type'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'quantity': pos['quantity'],
                'pnl': pnl,
                'balance': account_balance,
                'entry_time': pos['entry_time'],
                'exit_time': datetime.now()
            }

            closed_trades.append(trade_result)
            log_event('TRADE_CLOSED', trade_result)

            print(f"\n  [TRADE CLOSED] {pair} {pos['type']}")
            print(f"    Entry: ${pos['entry_price']:.5f} -> Exit: ${exit_price:.5f}")
            print(f"    P&L: ${pnl:+.2f} | Balance: ${account_balance:,.2f}")
            print(f"    Reason: {exit_reason}")

            del open_positions[pair]
            pair_cooldowns[pair] = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)

            # Reset state
            ps['state'] = 'waiting_for_F'
            ps['F_direction'] = None
            ps['sequence_closes'] = []
            return

    # ============================================
    # STATE MACHINE: WAITING FOR F (BREAKOUT)
    # ============================================

    if ps['state'] == 'waiting_for_F':
        if current_price > A:
            # F = Breakout ABOVE zone
            ps['state'] = 'waiting_for_D'
            ps['F_direction'] = 'above'
            ps['sequence_closes'] = [current_price]
            stats['F_signals'] += 1

            print(f"\n  [F SIGNAL] {pair} - Breakout ABOVE zone at ${current_price:.5f}")
            log_event('F_SIGNAL', {'pair': pair, 'direction': 'above', 'price': current_price})

        elif current_price < B:
            # F = Breakout BELOW zone
            ps['state'] = 'waiting_for_D'
            ps['F_direction'] = 'below'
            ps['sequence_closes'] = [current_price]
            stats['F_signals'] += 1

            print(f"\n  [F SIGNAL] {pair} - Breakout BELOW zone at ${current_price:.5f}")
            log_event('F_SIGNAL', {'pair': pair, 'direction': 'below', 'price': current_price})

    # ============================================
    # STATE MACHINE: WAITING FOR D (RE-ENTRY)
    # ============================================

    elif ps['state'] == 'waiting_for_D':
        ps['sequence_closes'].append(current_price)

        # Check if price is back inside zone
        if B <= current_price <= A:
            # D = Re-entry signal - THIS IS ENTRY
            stats['D_signals'] += 1

            D = round_price(current_price)

            if ps['F_direction'] == 'above':
                trade_type = 'SHORT'
                # 2% stop loss above entry, 2% take profit below entry
                S = round_price(D * (1 + STOP_LOSS_PCT))
                T = round_price(D * (1 - TAKE_PROFIT_PCT))
                distance = round_price(S - D)
            else:
                trade_type = 'LONG'
                # 2% stop loss below entry, 2% take profit above entry
                S = round_price(D * (1 - STOP_LOSS_PCT))
                T = round_price(D * (1 + TAKE_PROFIT_PCT))
                distance = round_price(D - S)

            print(f"\n  [D SIGNAL] {pair} - Re-entry at ${D:.5f}")
            print(f"    Type: {trade_type} | S: ${S:.5f} (2% SL) | T: ${T:.5f} (2% TP)")
            log_event('D_SIGNAL', {'pair': pair, 'type': trade_type, 'D': D, 'S': S, 'T': T})

            # Check if we can trade
            can_trade = True
            if pair in open_positions:
                can_trade = False
                print(f"    [SKIP] Already have position in {pair}")
            if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
                can_trade = False
                print(f"    [SKIP] Max positions reached ({MAX_CONCURRENT_POSITIONS})")
            if pair in pair_cooldowns and datetime.now() < pair_cooldowns[pair]:
                can_trade = False
                print(f"    [SKIP] Pair in cooldown")
            if distance <= 0:
                can_trade = False
                print(f"    [SKIP] Invalid distance")

            if can_trade:
                # Position sizing
                risk_amount = account_balance * RISK_PER_TRADE_PCT
                quantity = risk_amount / distance if distance > 0 else 0

                # Cap position value
                position_value = quantity * D
                if position_value > MAX_POSITION_VALUE:
                    quantity = MAX_POSITION_VALUE / D

                if quantity > 0:
                    entry_price = round_price(D * (1 + SLIPPAGE_PCT) if trade_type == 'LONG' else D * (1 - SLIPPAGE_PCT))

                    # Entry fee
                    fee = entry_price * quantity * FEE_PCT
                    account_balance -= fee

                    if trade_type == 'LONG':
                        stats['long_trades'] += 1
                    else:
                        stats['short_trades'] += 1

                    open_positions[pair] = {
                        'type': trade_type,
                        'entry_price': entry_price,
                        'D': D,
                        'S': S,
                        'T': T,
                        'A': A,
                        'B': B,
                        'quantity': quantity,
                        'entry_time': datetime.now(),
                        'distance': distance
                    }

                    print(f"\n  [TRADE OPENED] {pair} {trade_type}")
                    print(f"    Entry: ${entry_price:.5f}")
                    print(f"    Stop Loss (S): ${S:.5f}")
                    print(f"    Take Profit (T): ${T:.5f}")
                    print(f"    Quantity: {quantity:.6f}")
                    print(f"    Risk: ${risk_amount:.2f}")

                    log_event('TRADE_OPENED', {
                        'pair': pair,
                        'type': trade_type,
                        'entry_price': entry_price,
                        'S': S,
                        'T': T,
                        'quantity': quantity
                    })

            # Reset state
            ps['state'] = 'waiting_for_F'
            ps['F_direction'] = None
            ps['sequence_closes'] = []

def print_status():
    """Print current status"""
    print("\n" + "-"*50)
    print(f"[STATUS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Balance: ${account_balance:,.2f} (Peak: ${peak_balance:,.2f})")
    print(f"P&L: ${stats['total_pnl']:+,.2f}")
    print(f"Open Positions: {len(open_positions)}/{MAX_CONCURRENT_POSITIONS}")

    if open_positions:
        for pair, pos in open_positions.items():
            current = get_current_price(pair)
            if current:
                if pos['type'] == 'LONG':
                    unrealized = (current - pos['entry_price']) * pos['quantity']
                else:
                    unrealized = (pos['entry_price'] - current) * pos['quantity']
                print(f"  {pair} {pos['type']}: Entry ${pos['entry_price']:.5f} | Current ${current:.5f} | Unrealized: ${unrealized:+.2f}")

    print(f"Zones: {stats['zones_established']} | F: {stats['F_signals']} | D: {stats['D_signals']}")
    print(f"Trades: LONG {stats['long_trades']} | SHORT {stats['short_trades']}")
    print("-"*50)

# ============================================
# MAIN LOOP
# ============================================

def main():
    global pair_states

    print("\n[STARTING] Simulation bot starting...")
    print("[INFO] Press Ctrl+C to stop\n")

    # Initialize zones for all pairs
    print("[INIT] Establishing initial zones...")
    for pair in TRADING_PAIRS:
        print(f"  {pair}...", end=" ")
        zone = establish_zone(pair)
        if zone:
            pair_states[pair] = zone
            print(f"A=${zone['A']:.5f}, B=${zone['B']:.5f}")
        else:
            print("No valid zone")
        time.sleep(1)  # Rate limiting

    print(f"\n[READY] Monitoring {len(pair_states)} pairs")
    log_event('BOT_STARTED', {'pairs': list(pair_states.keys()), 'balance': STARTING_BALANCE})

    iteration = 0
    last_zone_update = datetime.now()

    try:
        while True:
            iteration += 1

            # Check if we need to re-establish zones (every 4 hours)
            if datetime.now() - last_zone_update > timedelta(hours=4):
                print("\n[ZONE UPDATE] Re-establishing 4-hour zones...")
                for pair in TRADING_PAIRS:
                    if pair not in open_positions:  # Don't update if we have a position
                        zone = establish_zone(pair)
                        if zone:
                            pair_states[pair] = zone
                            print(f"  {pair}: A=${zone['A']:.5f}, B=${zone['B']:.5f}")
                    time.sleep(0.5)
                last_zone_update = datetime.now()

            # Check each pair
            for pair in TRADING_PAIRS:
                if pair not in pair_states:
                    continue

                current_price = get_current_price(pair)
                if current_price:
                    check_signals(pair, current_price)

                time.sleep(0.3)  # Rate limiting

            # Print status every 5 iterations
            if iteration % 5 == 0:
                print_status()

            # Wait for next update
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n[STOPPING] Bot stopped by user")

        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
        print(f"Final Balance:    ${account_balance:,.2f}")
        print(f"Total P&L:        ${stats['total_pnl']:+,.2f}")
        print(f"ROI:              {(stats['total_pnl']/STARTING_BALANCE)*100:+.2f}%")
        print(f"\nTotal Trades: {len(closed_trades)}")
        print(f"LONG Trades:  {stats['long_trades']}")
        print(f"SHORT Trades: {stats['short_trades']}")

        if closed_trades:
            wins = len([t for t in closed_trades if t['pnl'] > 0])
            print(f"Win Rate:     {wins/len(closed_trades)*100:.1f}%")

        log_event('BOT_STOPPED', {
            'final_balance': account_balance,
            'total_pnl': stats['total_pnl'],
            'total_trades': len(closed_trades),
            'stats': stats
        })

        print("\nLog saved to simulation_log.json")
        print("="*70)

if __name__ == "__main__":
    main()
