#!/usr/bin/env python3
"""
LIVE TRADING: 5-Minute NYSE Opening Range Breakout Strategy
Starting Balance: $500
*** REAL MONEY - LIVE ORDERS ***
"""

import os
import time
import json
import pytz
import talib
import krakenex
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

print("\n" + "="*70)
print("*** LIVE TRADING MODE - REAL MONEY ***")
print("5-MINUTE NYSE OPENING RANGE BREAKOUT")
print("Starting at 9:00 AM ET (New York Time)")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================

STARTING_BALANCE = 300.0
MAX_CONCURRENT_POSITIONS = 5
RISK_PER_TRADE_PCT = 0.05

# Safety limits
CIRCUIT_BREAKER_LOSS = STARTING_BALANCE * 0.20  # $60
TRAILING_STOP_PCT = 0.15
DAILY_LOSS_LIMIT = STARTING_BALANCE * 0.05  # $15
MAX_POSITION_VALUE = STARTING_BALANCE * 0.30  # $90

# Fees
FEE_PCT = 0.0026  # Kraken taker fee
SLIPPAGE_PCT = 0.0005

# Strategy
SL_BUFFER = 0.02  # 2% stop loss
COOLDOWN_MINUTES = 3

# NYSE Opening Time (9:00 AM ET)
NYSE_OPEN_HOUR_ET = 9
NYSE_OPEN_MINUTE_ET = 0

# LIVE TRADING MODE
LIVE_TRADING_ENABLED = True

# Trading pairs (Kraken format) - Same as backtest
TRADING_PAIRS = [
    'AAVEUSD', 'ACHUSD', 'ADAUSD', 'ALCXUSD', 'ALGOUSD',
    'ANKRUSD', 'APEUSD', 'ATOMUSD', 'AVAXUSD', 'BATUSD',
    'BCHUSD', 'BONDUSD', 'COMPUSD', 'DOTUSD', 'ETHUSD',
    'HNTUSD', 'KNCUSD', 'LINKUSD', 'LQTYUSD', 'LTCUSD',
    'OMGUSD', 'PEPEUSD', 'POLUSD', 'QNTUSD', 'SCUSD',
    'SOLUSD', 'SUIUSD', 'TAOUSD', 'TRXUSD', 'UNIUSD',
    'XLMUSD', 'XMRUSD', 'XRPUSD', 'ZECUSD'
]

print(f"Starting Balance: ${STARTING_BALANCE:,.2f}")
print(f"Max Positions: {MAX_CONCURRENT_POSITIONS}")
print(f"Risk per Trade: {RISK_PER_TRADE_PCT*100}%")
print(f"Stop Loss: {SL_BUFFER*100}%")
print(f"Cooldown: {COOLDOWN_MINUTES} minutes")
print(f"NYSE Open Time: {NYSE_OPEN_HOUR_ET:02d}:{NYSE_OPEN_MINUTE_ET:02d} ET")

# ============================================
# KRAKEN API SETUP
# ============================================

api = krakenex.API()
api_key = os.getenv('KRAKEN_PUBLIC_KEY')
api_secret = os.getenv('KRAKEN_PRIVATE_KEY')

if not api_key or not api_secret:
    print("\n[ERROR] Kraken API keys not found in .env file!")
    exit(1)

api.key = api_key
api.secret = api_secret

print(f"\n[OK] Kraken API configured")
print(f"     Public Key: {api_key[:10]}...")

# ============================================
# STATE MANAGEMENT
# ============================================

state = {
    'account_balance': STARTING_BALANCE,
    'peak_balance': STARTING_BALANCE,
    'open_positions': {},
    'closed_trades': [],
    'daily_traded_pairs': set(),
    'pair_cooldowns': {},
    'current_day_pnl': 0,
    'opening_ranges': {},
    'is_market_open': False,
    'trading_date': None
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_current_time_et():
    """Get current time in New York (ET) timezone"""
    return datetime.now(pytz.timezone('America/New_York'))

def is_nyse_open_time():
    """Check if it's NYSE opening time (9:00-9:05 AM ET window)"""
    now_et = get_current_time_et()
    # Allow 5-minute window to establish opening range
    return now_et.hour == NYSE_OPEN_HOUR_ET and now_et.minute < 5

def get_ohlc_data(pair, interval=5, count=50):
    """Fetch OHLC data from Kraken"""
    try:
        result = api.query_public('OHLC', {
            'pair': pair,
            'interval': interval,
            'count': count
        })

        if result.get('error'):
            print(f"[ERROR] Kraken API error for {pair}: {result['error']}")
            return None

        # Parse OHLC data
        pair_key = list(result['result'].keys())[0]  # Kraken returns different pair names
        ohlc = result['result'][pair_key]

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

        return df

    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {pair}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI using TA-Lib"""
    try:
        if len(prices) < period:
            return None
        rsi = talib.RSI(np.array(prices), timeperiod=period)
        return rsi[-1] if len(rsi) > 0 else None
    except:
        return None

def place_order(pair, side, quantity, order_type='market'):
    """Place order on Kraken (LIVE TRADING)"""
    if not LIVE_TRADING_ENABLED:
        print(f"\n[PAPER TRADE] Would place {side} order:")
        print(f"  Pair: {pair}, Quantity: {quantity}")
        return True

    try:
        print(f"\n[LIVE ORDER] Placing {side} order on Kraken...")
        print(f"  Pair: {pair}")
        print(f"  Quantity: {quantity:.8f}")
        print(f"  Type: {order_type}")

        # Prepare order parameters
        order_params = {
            'pair': pair,
            'type': side.lower(),  # 'buy' or 'sell'
            'ordertype': order_type,
            'volume': str(quantity)
        }

        # Place order via Kraken API
        result = api.query_private('AddOrder', order_params)

        if result.get('error'):
            print(f"  [ERROR] Order failed: {result['error']}")
            return False

        order_id = result['result']['txid'][0] if result.get('result', {}).get('txid') else 'unknown'
        print(f"  [SUCCESS] Order placed! ID: {order_id}")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to place order: {e}")
        return False

def save_state():
    """Save trading state to JSON"""
    state_copy = state.copy()
    state_copy['daily_traded_pairs'] = list(state_copy['daily_traded_pairs'])

    with open('live_trading_state.json', 'w') as f:
        json.dump(state_copy, f, indent=2, default=str)

def log_trade(trade_info):
    """Log trade to file"""
    with open('live_trades_log.json', 'a') as f:
        f.write(json.dumps(trade_info, default=str) + '\n')

# ============================================
# TRADING LOGIC
# ============================================

def establish_opening_range():
    """Establish opening range using completed 9:00 AM ET candle (closes at 9:05 AM)"""
    now_et = get_current_time_et()
    print(f"\n[{now_et.strftime('%H:%M:%S')}] Establishing opening range...")

    # Determine which day's 9:00 AM candle to use
    target_date = now_et.date()

    # If before 9:05 AM ET (candle not closed yet), use previous day's candle
    if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 5):
        target_date = target_date - timedelta(days=1)
        print(f"  Before 9:05 AM ET - using previous day's breakout zone: {target_date}")
    else:
        print(f"  Using today's breakout zone: {target_date}")

    # Target: 9:00 AM candle (which closes at 9:05 AM)
    target_time_et = now_et.replace(hour=NYSE_OPEN_HOUR_ET, minute=NYSE_OPEN_MINUTE_ET, second=0, microsecond=0)
    if target_date != now_et.date():
        target_time_et = target_time_et - timedelta(days=1)

    print(f"  Target candle: {target_time_et.strftime('%Y-%m-%d %H:%M:%S ET')} (9:00 AM candle)")

    for pair in TRADING_PAIRS:
        # Fetch enough 5-minute candles to find the 9:00 AM candle
        candles_needed = max(150, int((now_et - target_time_et).total_seconds() / 300) + 10)
        df = get_ohlc_data(pair, interval=5, count=min(candles_needed, 720))  # Max 720 = 60 hours

        if df is not None and len(df) > 0:
            # Find the candle that starts at 9:00 AM ET
            df['time_diff'] = abs((df['timestamp'] - target_time_et).dt.total_seconds())
            closest_idx = df['time_diff'].idxmin()
            candle = df.loc[closest_idx]

            state['opening_ranges'][pair] = {
                'high': candle['high'],
                'low': candle['low'],
                'timestamp': str(candle['timestamp'])
            }
            print(f"  {pair}: High={candle['high']:.4f}, Low={candle['low']:.4f}")

    state['is_market_open'] = True
    state['trading_date'] = target_date
    save_state()

def check_for_signals():
    """Check for breakout signals"""
    now_et = get_current_time_et()

    for pair in TRADING_PAIRS:
        # Skip if already in position
        if pair in state['open_positions']:
            continue

        # Skip if at max positions
        if len(state['open_positions']) >= MAX_CONCURRENT_POSITIONS:
            continue

        # Skip if already traded today
        if pair in state['daily_traded_pairs']:
            continue

        # Check cooldown
        if pair in state['pair_cooldowns']:
            if now_et < state['pair_cooldowns'][pair]:
                continue
            else:
                del state['pair_cooldowns'][pair]

        # Get opening range
        if pair not in state['opening_ranges']:
            continue

        orb = state['opening_ranges'][pair]

        # Get current price
        df = get_ohlc_data(pair, interval=5, count=20)
        if df is None or len(df) < 20:
            continue

        current_candle = df.iloc[-1]
        close_price = current_candle['close']

        # Check for breakout
        signal_type = None
        if close_price > orb['high']:
            signal_type = "LONG"
        elif close_price < orb['low']:
            signal_type = "SHORT"

        if signal_type:
            enter_position(pair, signal_type, close_price, orb, df)

def enter_position(pair, signal_type, close_price, orb, historical_data):
    """Enter a new position"""
    # Calculate entry with slippage
    if signal_type == "LONG":
        entry_price = close_price * (1 + SLIPPAGE_PCT)
        stop_loss = entry_price * (1 - SL_BUFFER)
        risk_per_unit = entry_price - stop_loss
        take_profit = entry_price + (2 * risk_per_unit)
    else:
        entry_price = close_price * (1 - SLIPPAGE_PCT)
        stop_loss = entry_price * (1 + SL_BUFFER)
        risk_per_unit = stop_loss - entry_price
        take_profit = entry_price - (2 * risk_per_unit)

    if risk_per_unit <= 0:
        return

    # Position sizing
    risk_amount = state['account_balance'] * RISK_PER_TRADE_PCT
    quantity = risk_amount / risk_per_unit

    # Safety cap
    position_value = quantity * entry_price
    if position_value > MAX_POSITION_VALUE:
        quantity = MAX_POSITION_VALUE / entry_price

    # Calculate RSI for tracking
    rsi = calculate_rsi(historical_data['close'].values)

    # Store position
    state['open_positions'][pair] = {
        'type': signal_type,
        'entry_price': entry_price,
        'entry_time': get_current_time_et(),
        'sl': stop_loss,
        'tp': take_profit,
        'quantity': quantity,
        'rsi': rsi if rsi else 0
    }

    state['daily_traded_pairs'].add(pair)

    print(f"\n[TRADE] {signal_type} {pair}")
    print(f"  Entry: ${entry_price:.4f}")
    print(f"  Quantity: {quantity:.6f}")
    print(f"  SL: ${stop_loss:.4f} (-{SL_BUFFER*100}%)")
    print(f"  TP: ${take_profit:.4f} (+{SL_BUFFER*2*100}%)")
    print(f"  Risk: ${quantity * risk_per_unit:.2f}")

    # Place order (live trading)
    order_success = place_order(pair, 'BUY' if signal_type == 'LONG' else 'SELL', quantity)

    if not order_success:
        print(f"  [WARNING] Order failed - position not tracked")
        return

    save_state()
    log_trade({
        'action': 'ENTER',
        'pair': pair,
        'type': signal_type,
        'entry_price': entry_price,
        'quantity': quantity,
        'sl': stop_loss,
        'tp': take_profit,
        'timestamp': get_current_time_et()
    })

def check_exits():
    """Check if any positions need to exit"""
    pairs_to_close = []

    for pair, position in state['open_positions'].items():
        df = get_ohlc_data(pair, interval=5, count=1)

        if df is None or len(df) == 0:
            continue

        candle = df.iloc[-1]
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
            exit_position(pair, position, exit_price, exit_reason)
            pairs_to_close.append(pair)

    for pair in pairs_to_close:
        del state['open_positions'][pair]

def exit_position(pair, position, exit_price, exit_reason):
    """Exit a position"""
    if position['type'] == 'LONG':
        pnl = (exit_price - position['entry_price']) * position['quantity']
    else:
        pnl = (position['entry_price'] - exit_price) * position['quantity']

    # Deduct fees
    total_fees = (position['entry_price'] * position['quantity'] * FEE_PCT) + \
                 (exit_price * position['quantity'] * FEE_PCT)
    pnl -= total_fees

    # Update balance
    state['account_balance'] += pnl
    state['current_day_pnl'] += pnl

    if state['account_balance'] > state['peak_balance']:
        state['peak_balance'] = state['account_balance']

    # Set cooldown
    cooldown_end = get_current_time_et() + timedelta(minutes=COOLDOWN_MINUTES)
    state['pair_cooldowns'][pair] = cooldown_end

    print(f"\n[EXIT] {exit_reason} - {pair}")
    print(f"  Exit: ${exit_price:.4f}")
    print(f"  P&L: ${pnl:+.2f}")
    print(f"  Balance: ${state['account_balance']:.2f}")

    # Place exit order (live trading)
    place_order(pair, 'SELL' if position['type'] == 'LONG' else 'BUY', position['quantity'])

    trade_info = {
        'action': 'EXIT',
        'pair': pair,
        'type': position['type'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'quantity': position['quantity'],
        'pnl': pnl,
        'balance': state['account_balance'],
        'timestamp': get_current_time_et()
    }

    state['closed_trades'].append(trade_info)
    save_state()
    log_trade(trade_info)

# ============================================
# MAIN LOOP
# ============================================

def main():
    print("\n[STARTING] Live trading bot...")

    if LIVE_TRADING_ENABLED:
        print("\n" + "!"*70)
        print("!!! WARNING: LIVE TRADING MODE - REAL MONEY !!!")
        print("!"*70)
        print(f"\nStarting balance: ${STARTING_BALANCE}")
        print(f"Circuit breaker: ${CIRCUIT_BREAKER_LOSS} loss")
        print(f"Risk per trade: 5% (${STARTING_BALANCE * RISK_PER_TRADE_PCT})")
        print("\nPress Ctrl+C within 10 seconds to cancel...")

        try:
            for i in range(10, 0, -1):
                print(f"  Starting in {i}...", end='\r')
                time.sleep(1)
            print("\n\n[LIVE] Bot activated - placing REAL orders\n")
        except KeyboardInterrupt:
            print("\n\n[CANCELLED] Bot stopped by user")
            return
    else:
        print("[MODE] PAPER TRADING - No real orders will be placed")

    print("[INFO] Bot will trade 24/7 using daily 9:00 AM ET opening range\n")

    # Establish opening range on startup (will find today's 9:00 AM candle)
    establish_opening_range()

    while True:
        try:
            now_et = get_current_time_et()

            # Check if we need to establish a NEW opening range (at 9:05 AM ET after candle closes)
            if state['trading_date'] and now_et.date() != state['trading_date']:
                # New day detected - check if it's past 9:05 AM ET (candle has closed)
                if now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 5):
                    print(f"\n[NEW DAY] 9:00 AM candle closed - establishing new opening range for {now_et.date()}")
                    state['daily_traded_pairs'] = set()
                    state['current_day_pnl'] = 0
                    state['opening_ranges'] = {}
                    establish_opening_range()

            # Trade 24/7 using current opening range (crypto markets never close)
            if state['is_market_open']:
                # Check exits every minute
                check_exits()

                # Check for new signals every minute
                check_for_signals()

                # Safety checks
                total_pnl = state['account_balance'] - STARTING_BALANCE
                if total_pnl <= -CIRCUIT_BREAKER_LOSS:
                    print(f"\n[CIRCUIT BREAKER] Daily loss limit hit: ${total_pnl:.2f}")
                    break

                drawdown = (state['peak_balance'] - state['account_balance']) / state['peak_balance']
                if drawdown >= TRAILING_STOP_PCT:
                    print(f"\n[TRAILING STOP] Max drawdown hit: {drawdown*100:.1f}%")
                    break

            # Sleep for 1 minute
            time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n[STOPPING] Bot stopped by user")
            save_state()
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            save_state()
            time.sleep(60)

if __name__ == "__main__":
    main()
