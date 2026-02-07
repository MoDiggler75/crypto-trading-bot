#!/usr/bin/env python3
"""
Trading Bot Dashboard - Web-based monitoring for simulation bots.

Reads state files written by both simulation bots and displays
real-time progress via a browser-based dashboard.

Usage: python3 dashboard.py
Then open http://localhost:5000 in your browser.
"""

import json
import os
from datetime import datetime
from flask import Flask, render_template_string

app = Flask(__name__)

STATE_FILES = {
    '4hr': 'sim_4hr_state.json',
    '5min': 'sim_5min_state.json',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_bot_state(key):
    """Load a bot's state from its JSON file."""
    path = os.path.join(BASE_DIR, STATE_FILES[key])
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="10">
<title>Trading Bot Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d28;
    --border: #2a2d3a;
    --text: #e1e4ea;
    --muted: #8b8fa3;
    --green: #00c853;
    --red: #ff1744;
    --blue: #2979ff;
    --amber: #ffc400;
    --purple: #aa00ff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg);
    color: var(--text);
    padding: 20px;
    min-height: 100vh;
  }
  .header {
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .header h1 { font-size: 1.5rem; color: var(--blue); letter-spacing: 2px; }
  .header .time { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .bot-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }
  .bot-card.offline { opacity: 0.5; }
  .bot-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
  }
  .bot-name { font-size: 1.1rem; font-weight: bold; }
  .status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    padding: 4px 12px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .status.running { background: rgba(0,200,83,0.15); color: var(--green); }
  .status.stopped { background: rgba(255,23,68,0.15); color: var(--red); }
  .status.offline { background: rgba(139,143,163,0.15); color: var(--muted); }
  .status .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
  }
  .status.running .dot { animation: pulse 1.5s infinite; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 20px;
  }
  .metric {
    background: rgba(255,255,255,0.03);
    padding: 12px;
    border-radius: 8px;
  }
  .metric .label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .metric .value { font-size: 1.2rem; margin-top: 4px; font-weight: bold; }
  .metric .value.positive { color: var(--green); }
  .metric .value.negative { color: var(--red); }
  .section-title {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 16px 0 10px 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }
  th {
    text-align: left;
    color: var(--muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
  }
  .long { color: var(--green); }
  .short { color: var(--red); }
  .tp { color: var(--green); }
  .sl { color: var(--red); }
  .zone-bar {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    margin-top: 6px;
    position: relative;
  }
  .zone-bar .fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--blue), var(--purple));
  }
  .empty-msg {
    color: var(--muted);
    font-size: 0.8rem;
    text-align: center;
    padding: 16px 0;
  }
  .footer {
    text-align: center;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.75rem;
  }
  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>CRYPTO TRADING BOT DASHBOARD</h1>
  <div class="time">Last refresh: {{ now }} &mdash; Auto-refreshes every 10s</div>
</div>

<div class="grid">
{% for bot in bots %}
  <div class="bot-card {{ 'offline' if bot.status == 'offline' else '' }}">
    <div class="bot-header">
      <span class="bot-name">{{ bot.name }}</span>
      <span class="status {{ bot.status }}"><span class="dot"></span> {{ bot.status }}</span>
    </div>

    <div class="metrics">
      <div class="metric">
        <div class="label">Balance</div>
        <div class="value">${{ "%.2f"|format(bot.balance) }}</div>
      </div>
      <div class="metric">
        <div class="label">P&amp;L</div>
        <div class="value {{ 'positive' if bot.pnl >= 0 else 'negative' }}">
          {{ "%+.2f"|format(bot.pnl) }} ({{ "%+.1f"|format(bot.roi) }}%)
        </div>
      </div>
      <div class="metric">
        <div class="label">Peak Balance</div>
        <div class="value">${{ "%.2f"|format(bot.peak) }}</div>
      </div>
      <div class="metric">
        <div class="label">Win Rate</div>
        <div class="value">{{ "%.1f"|format(bot.win_rate) }}%
          <span style="font-size:0.7rem;color:var(--muted);">({{ bot.wins }}W / {{ bot.losses }}L)</span>
        </div>
      </div>
      <div class="metric">
        <div class="label">Total Trades</div>
        <div class="value">{{ bot.total_trades }}
          <span style="font-size:0.7rem;color:var(--muted);">({{ bot.longs }}L / {{ bot.shorts }}S)</span>
        </div>
      </div>
      <div class="metric">
        <div class="label">Last Update</div>
        <div class="value" style="font-size:0.85rem;">{{ bot.last_update }}</div>
      </div>
    </div>

    <div class="section-title">Open Positions ({{ bot.open_positions|length }}/{{ bot.max_positions }})</div>
    {% if bot.open_positions %}
    <table>
      <tr><th>Pair</th><th>Type</th><th>Entry</th><th>SL</th><th>TP</th></tr>
      {% for pos in bot.open_positions %}
      <tr>
        <td>{{ pos.pair }}</td>
        <td class="{{ pos.type|lower }}">{{ pos.type }}</td>
        <td>${{ "%.2f"|format(pos.entry_price) if pos.entry_price > 100 else "%.5f"|format(pos.entry_price) }}</td>
        <td class="sl">${{ "%.2f"|format(pos.stop_loss) if pos.stop_loss > 100 else "%.5f"|format(pos.stop_loss) }}</td>
        <td class="tp">${{ "%.2f"|format(pos.take_profit) if pos.take_profit > 100 else "%.5f"|format(pos.take_profit) }}</td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty-msg">No open positions</div>
    {% endif %}

    <div class="section-title">Active Zones</div>
    {% if bot.zones %}
    <table>
      <tr><th>Pair</th><th>High / Low</th><th>Range</th></tr>
      {% for z in bot.zones %}
      <tr>
        <td>{{ z.pair }}</td>
        <td>${{ "%.2f"|format(z.high) if z.high > 100 else "%.5f"|format(z.high) }}
            / ${{ "%.2f"|format(z.low) if z.low > 100 else "%.5f"|format(z.low) }}</td>
        <td>${{ "%.2f"|format(z.range) if z.range > 1 else "%.5f"|format(z.range) }}</td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty-msg">No active zones</div>
    {% endif %}

    <div class="section-title">Recent Trades</div>
    {% if bot.closed_trades %}
    <table>
      <tr><th>Pair</th><th>Type</th><th>P&amp;L</th><th>Exit</th></tr>
      {% for trade in bot.closed_trades[-8:]|reverse %}
      <tr>
        <td>{{ trade.pair }}</td>
        <td class="{{ trade.type|lower }}">{{ trade.type }}</td>
        <td class="{{ 'positive' if trade.pnl >= 0 else 'negative' }}">
          ${{ "%+.2f"|format(trade.pnl) }}
        </td>
        <td class="{{ 'tp' if trade.exit_reason == 'TAKE_PROFIT' else 'sl' }}">
          {{ trade.exit_reason }}
        </td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty-msg">No closed trades yet</div>
    {% endif %}

  </div>
{% endfor %}
</div>

<div class="footer">
  Simulation bots &mdash; Paper trading only, no real money at risk
</div>

</body>
</html>
"""


def build_bot_data(state, default_name, max_pos=3):
    """Transform raw state dict into template-friendly data."""
    if state is None:
        return {
            'name': default_name,
            'status': 'offline',
            'balance': 0, 'pnl': 0, 'roi': 0, 'peak': 0,
            'win_rate': 0, 'wins': 0, 'losses': 0,
            'total_trades': 0, 'longs': 0, 'shorts': 0,
            'max_positions': max_pos,
            'last_update': 'N/A',
            'open_positions': [],
            'closed_trades': [],
            'zones': [],
        }

    s = state.get('stats', {})

    # Build zone list
    zones = []
    for pair, z in state.get('active_zones', {}).items():
        high = z.get('high') or z.get('A', 0)
        low = z.get('low') or z.get('B', 0)
        zones.append({
            'pair': pair,
            'high': high,
            'low': low,
            'range': z.get('range', high - low),
        })

    last = state.get('last_update', '')
    if last:
        try:
            dt = datetime.fromisoformat(last)
            last = dt.strftime('%H:%M:%S')
        except Exception:
            pass

    return {
        'name': state.get('bot_name', default_name),
        'status': state.get('status', 'offline'),
        'balance': state.get('balance', 0),
        'pnl': state.get('total_pnl', 0),
        'roi': state.get('roi_pct', 0),
        'peak': state.get('peak_balance', 0),
        'win_rate': s.get('win_rate', 0),
        'wins': s.get('wins', 0),
        'losses': s.get('losses', 0),
        'total_trades': s.get('total_trades', 0),
        'longs': s.get('long_trades', 0),
        'shorts': s.get('short_trades', 0),
        'max_positions': max_pos,
        'last_update': last,
        'open_positions': state.get('open_positions', []),
        'closed_trades': state.get('closed_trades', []),
        'zones': zones,
    }


@app.route('/')
def dashboard():
    state_4hr = load_bot_state('4hr')
    state_5min = load_bot_state('5min')

    bots = [
        build_bot_data(state_4hr, '4-Hour Breakout Zone Retest'),
        build_bot_data(state_5min, '5-Min NYSE Opening Range Breakout'),
    ]

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return render_template_string(DASHBOARD_HTML, bots=bots, now=now)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("TRADING BOT DASHBOARD")
    print("Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
