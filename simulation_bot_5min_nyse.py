#!/usr/bin/env python3
"""
SIMULATION BOT - 5-Minute NYSE Opening Range Breakout Strategy
Uses live Kraken data but only paper trades (no real money)

Strategy:
1. At 9:00 AM ET (NYSE open), captures the first 5-min candle (9:00-9:05)
2. Uses high/low of that candle as the breakout zone for the day
3. Breakout above zone high -> LONG entry
4. Breakout below zone low -> SHORT entry
5. Stop loss at opposite zone boundary
6. Take profit at 2x zone range from entry
7. Zones reset daily at next 9:00 AM ET open
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
import pytz
from dotenv import load_dotenv

load_dotenv()

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("\n" + "=" * 70)
print("SIMULATION BOT - 5-Min NYSE Opening Range Breakout")
print("MODE: PAPER TRADING (No real money)")
print("=" * 70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

STARTING_BALANCE = 10000.0
account_balance = STARTING_BALANCE
peak_balance = STARTING_BALANCE

MAX_CONCURRENT_POSITIONS = 3
RISK_PER_TRADE_PCT = 0.05
MAX_POSITION_VALUE = STARTING_BALANCE * 0.30

FEE_PCT = 0.0026
SLIPPAGE_PCT = 0.0005

PRICE_DECIMALS = 5

COOLDOWN_MINUTES = 15

# NYSE Opening Time
NYSE_OPEN_HOUR = 9
NYSE_OPEN_MINUTE = 0
ET_TIMEZONE = pytz.timezone('US/Eastern')

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

UPDATE_INTERVAL = 60

LOG_FILE = 'simulation_5min_log.json'
STATE_FILE = 'sim_5min_state.json'

print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
print(f"Max Positions: {MAX_CONCURRENT_POSITIONS}")
print(f"Risk per Trade: {RISK_PER_TRADE_PCT * 100}%")
print(f"Trading Pairs: {len(TRADING_PAIRS)}")
print(f"Update Interval: {UPDATE_INTERVAL}s")
print(f"Strategy: First 5-min candle at 9:00 AM ET")

# ============================================
# HELPER FUNCTIONS
# ============================================

def round_price(price):
    return round(float(price), PRICE_DECIMALS)


def log_event(event_type, data):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': event_type,
        'data': data
    }
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(log_entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2, default=str)
    return log_entry


def write_state():
    """Write current state for dashboard consumption."""
    wins = len([t for t in closed_trades if t['pnl'] > 0])
    losses = len([t for t in closed_trades if t['pnl'] <= 0])
    total = wins + losses
    state = {
        'bot_name': '5-Min NYSE Opening Range Breakout',
        'strategy': '5min_nyse_orb',
        'status': 'running',
        'last_update': datetime.now().isoformat(),
        'balance': account_balance,
        'starting_balance': STARTING_BALANCE,
        'peak_balance': peak_balance,
        'total_pnl': stats['total_pnl'],
        'roi_pct': (stats['total_pnl'] / STARTING_BALANCE) * 100,
        'open_positions': [],
        'closed_trades': [],
        'stats': {
            'zones_established': stats['zones_established'],
            'breakout_signals': stats['breakout_signals'],
            'long_trades': stats['long_trades'],
            'short_trades': stats['short_trades'],
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total * 100) if total > 0 else 0,
        },
        'active_zones': {}
    }
    for pair, pos in open_positions.items():
        state['open_positions'].append({
            'pair': pair,
            'type': pos['type'],
            'entry_price': pos['entry_price'],
            'stop_loss': pos['S'],
            'take_profit': pos['T'],
            'quantity': pos['quantity'],
            'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else str(pos['entry_time'])
        })
    for trade in closed_trades[-20:]:
        state['closed_trades'].append({
            'pair': trade['pair'],
            'type': trade['type'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'pnl': trade['pnl'],
            'exit_reason': trade['exit_reason'],
            'entry_time': trade['entry_time'].isoformat() if isinstance(trade['entry_time'], datetime) else str(trade['entry_time']),
            'exit_time': trade['exit_time'].isoformat() if isinstance(trade['exit_time'], datetime) else str(trade['exit_time'])
        })
    for pair, zone in daily_zones.items():
        state['active_zones'][pair] = {
            'high': zone['high'],
            'low': zone['low'],
            'range': zone['range'],
            'date': zone['date'],
            'established_at': zone['established_at'].isoformat() if isinstance(zone['established_at'], datetime) else str(zone['established_at'])
        }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ============================================
# KRAKEN API SETUP
# ============================================

print("\n[SETUP] Connecting to Kraken API...")

try:
    api = krakenex.API()
    kraken = KrakenAPI(api)
    print("[OK] Connected to Kraken API (public data only)")
except Exception as e:
    print(f"[ERROR] Failed to connect: {e}")
    exit(1)

# ============================================
# STATE MANAGEMENT
# ============================================

daily_zones = {}          # {pair: {high, low, range, date, established_at}}
open_positions = {}
closed_trades = []
pair_cooldowns = {}
current_zone_date = None  # Track which day's zone we have

stats = {
    'zones_established': 0,
    'breakout_signals': 0,
    'long_trades': 0,
    'short_trades': 0,
    'total_pnl': 0.0
}

# ============================================
# STRATEGY FUNCTIONS
# ============================================

def get_current_time_et():
    """Get current time in Eastern Time."""
    return datetime.now(ET_TIMEZONE)


def get_opening_candle(pair):
    """
    Get the 5-minute candle starting at 9:00 AM ET today.
    Returns the candle data or None if not yet available.
    """
    now_et = get_current_time_et()
    today_open = now_et.replace(hour=NYSE_OPEN_HOUR, minute=NYSE_OPEN_MINUTE, second=0, microsecond=0)

    # Opening candle runs 9:00 - 9:05. Wait until it's complete.
    if now_et < today_open + timedelta(minutes=5):
        return None

    try:
        ohlc, _ = kraken.get_ohlc_data(pair, interval=5, ascending=True)
        if len(ohlc) == 0:
            return None

        # ohlc index is a DatetimeIndex (UTC). Find the candle at 9:00 AM ET today.
        target_utc = today_open.astimezone(pytz.utc)

        for idx in ohlc.index:
            candle_time = idx
            if hasattr(candle_time, 'tz') and candle_time.tz is None:
                candle_time = candle_time.tz_localize('UTC')

            # Allow 60-second tolerance for matching
            diff = abs((candle_time - target_utc).total_seconds())
            if diff < 60:
                row = ohlc.loc[idx]
                return {
                    'open': round_price(row['open']),
                    'high': round_price(row['high']),
                    'low': round_price(row['low']),
                    'close': round_price(row['close']),
                    'time': idx
                }
    except Exception as e:
        print(f"  [ERROR] {pair} opening candle: {e}")

    return None


def establish_daily_zone(pair):
    """Establish the opening range zone from the first 5-min candle at NYSE open."""
    candle = get_opening_candle(pair)
    if candle is None:
        return None

    high = candle['high']
    low = candle['low']
    zone_range = high - low

    # Skip if range is too small (less than 0.01%)
    if zone_range / low < 0.0001:
        print(f"  {pair}: Zone range too small (${zone_range:.5f}), skipping")
        return None

    now_et = get_current_time_et()
    zone = {
        'high': high,
        'low': low,
        'range': zone_range,
        'date': now_et.strftime('%Y-%m-%d'),
        'established_at': datetime.now(),
        'candle_time': candle['time'],
        'breakout_triggered': False
    }

    stats['zones_established'] += 1
    log_event('ZONE_ESTABLISHED', {
        'pair': pair, 'high': high, 'low': low,
        'range': zone_range, 'date': zone['date']
    })

    return zone


def get_current_price(pair):
    """Get current ticker price."""
    try:
        ticker = kraken.get_ticker_information(pair)
        if pair in ticker.index:
            price = round_price(ticker.loc[pair, 'c'][0])
            return price
    except Exception as e:
        print(f"  [ERROR] {pair} ticker: {e}")
    return None


def check_breakout_and_manage(pair, current_price):
    """Check for breakout entries and manage open positions."""
    global account_balance, peak_balance

    if pair not in daily_zones:
        return

    zone = daily_zones[pair]
    high = zone['high']
    low = zone['low']
    zone_range = zone['range']

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
            return

    # ============================================
    # CHECK FOR BREAKOUT ENTRY
    # ============================================

    if pair in open_positions:
        return  # Already have a position

    if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
        return

    if pair in pair_cooldowns and datetime.now() < pair_cooldowns[pair]:
        return

    trade_type = None

    if current_price > high:
        # Breakout above opening range -> LONG
        trade_type = 'LONG'
        entry_price = round_price(current_price * (1 + SLIPPAGE_PCT))
        S = round_price(low)           # Stop loss at bottom of zone
        T = round_price(high + 2 * zone_range)  # Take profit at 2x range above zone high
        distance = round_price(entry_price - S)
    elif current_price < low:
        # Breakout below opening range -> SHORT
        trade_type = 'SHORT'
        entry_price = round_price(current_price * (1 - SLIPPAGE_PCT))
        S = round_price(high)          # Stop loss at top of zone
        T = round_price(low - 2 * zone_range)   # Take profit at 2x range below zone low
        distance = round_price(S - entry_price)

    if trade_type and distance > 0:
        stats['breakout_signals'] += 1

        risk_amount = account_balance * RISK_PER_TRADE_PCT
        quantity = risk_amount / distance

        position_value = quantity * entry_price
        if position_value > MAX_POSITION_VALUE:
            quantity = MAX_POSITION_VALUE / entry_price

        if quantity > 0:
            fee = entry_price * quantity * FEE_PCT
            account_balance -= fee

            if trade_type == 'LONG':
                stats['long_trades'] += 1
            else:
                stats['short_trades'] += 1

            open_positions[pair] = {
                'type': trade_type,
                'entry_price': entry_price,
                'S': S,
                'T': T,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'distance': distance,
                'zone_high': high,
                'zone_low': low
            }

            print(f"\n  [BREAKOUT {trade_type}] {pair}")
            print(f"    Zone: ${low:.5f} - ${high:.5f} (range: ${zone_range:.5f})")
            print(f"    Entry: ${entry_price:.5f}")
            print(f"    Stop Loss (S): ${S:.5f}")
            print(f"    Take Profit (T): ${T:.5f}")
            print(f"    Quantity: {quantity:.6f} | Risk: ${risk_amount:.2f}")

            log_event('TRADE_OPENED', {
                'pair': pair, 'type': trade_type,
                'entry_price': entry_price, 'S': S, 'T': T,
                'quantity': quantity, 'zone_high': high, 'zone_low': low
            })


def print_status():
    """Print current status."""
    now_et = get_current_time_et()
    print("\n" + "-" * 50)
    print(f"[STATUS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (ET: {now_et.strftime('%H:%M')})")
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
                print(f"  {pair} {pos['type']}: Entry ${pos['entry_price']:.5f} | "
                      f"Current ${current:.5f} | Unrealized: ${unrealized:+.2f}")

    print(f"Zones: {stats['zones_established']} | "
          f"Breakouts: {stats['breakout_signals']}")
    print(f"Trades: LONG {stats['long_trades']} | SHORT {stats['short_trades']}")

    if daily_zones:
        print("Active Zones:")
        for pair, zone in daily_zones.items():
            print(f"  {pair}: ${zone['low']:.5f} - ${zone['high']:.5f} "
                  f"(range: ${zone['range']:.5f})")

    print("-" * 50)


# ============================================
# MAIN LOOP
# ============================================

def main():
    global current_zone_date, daily_zones

    print("\n[STARTING] 5-Min NYSE Opening Range simulation starting...")
    print("[INFO] Press Ctrl+C to stop\n")
    print("[INFO] Waiting for 9:00 AM ET opening candle to establish zones...\n")

    log_event('BOT_STARTED', {
        'pairs': TRADING_PAIRS,
        'balance': STARTING_BALANCE,
        'strategy': '5min_nyse_opening_range_breakout'
    })

    iteration = 0

    try:
        while True:
            iteration += 1
            now_et = get_current_time_et()
            today_str = now_et.strftime('%Y-%m-%d')

            # ============================================
            # ESTABLISH ZONES AT NYSE OPEN
            # ============================================

            # Check if we need to establish new zones for today
            if current_zone_date != today_str:
                # Past 9:05 AM ET? Try to get the opening candle.
                nyse_open = now_et.replace(hour=NYSE_OPEN_HOUR, minute=NYSE_OPEN_MINUTE,
                                           second=0, microsecond=0)

                if now_et >= nyse_open + timedelta(minutes=5):
                    print(f"\n[ZONE UPDATE] Establishing opening range zones for {today_str}...")
                    zones_set = 0
                    for pair in TRADING_PAIRS:
                        if pair not in open_positions:
                            print(f"  {pair}...", end=" ")
                            zone = establish_daily_zone(pair)
                            if zone:
                                daily_zones[pair] = zone
                                print(f"High=${zone['high']:.5f}, Low=${zone['low']:.5f}, "
                                      f"Range=${zone['range']:.5f}")
                                zones_set += 1
                            else:
                                print("No valid zone")
                        else:
                            print(f"  {pair}... Skipped (open position)")
                        time.sleep(1)

                    if zones_set > 0:
                        current_zone_date = today_str
                        print(f"\n[READY] {zones_set} zones established for {today_str}")
                    else:
                        print("\n[WAIT] No zones established yet, will retry next iteration")

            # ============================================
            # CHECK BREAKOUTS AND MANAGE POSITIONS
            # ============================================

            for pair in TRADING_PAIRS:
                if pair not in daily_zones and pair not in open_positions:
                    continue

                current_price = get_current_price(pair)
                if current_price:
                    check_breakout_and_manage(pair, current_price)

                time.sleep(0.3)

            # Write state for dashboard
            write_state()

            # Print status every 5 iterations
            if iteration % 5 == 0:
                print_status()

            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n[STOPPING] Bot stopped by user")

        print("\n" + "=" * 70)
        print("FINAL SUMMARY - 5-Min NYSE Opening Range Breakout")
        print("=" * 70)
        print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
        print(f"Final Balance:    ${account_balance:,.2f}")
        print(f"Total P&L:        ${stats['total_pnl']:+,.2f}")
        print(f"ROI:              {(stats['total_pnl'] / STARTING_BALANCE) * 100:+.2f}%")
        print(f"\nTotal Trades: {len(closed_trades)}")
        print(f"LONG Trades:  {stats['long_trades']}")
        print(f"SHORT Trades: {stats['short_trades']}")

        if closed_trades:
            wins = len([t for t in closed_trades if t['pnl'] > 0])
            print(f"Win Rate:     {wins / len(closed_trades) * 100:.1f}%")

        # Write final state as stopped
        try:
            with open(STATE_FILE, 'r') as f:
                final_state = json.load(f)
            final_state['status'] = 'stopped'
            with open(STATE_FILE, 'w') as f:
                json.dump(final_state, f, indent=2, default=str)
        except Exception:
            pass

        log_event('BOT_STOPPED', {
            'final_balance': account_balance,
            'total_pnl': stats['total_pnl'],
            'total_trades': len(closed_trades),
            'stats': stats
        })

        print(f"\nLog saved to {LOG_FILE}")
        print("=" * 70)


if __name__ == "__main__":
    main()
