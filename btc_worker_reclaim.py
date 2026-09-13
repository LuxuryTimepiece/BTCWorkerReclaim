#!/usr/bin/env python3
"""
BTC Reclaim / Early-Trend Worker
================================
Philosophy: Catch the START of the bounce inside a friendly daily regime.
- Does NOT require full 4H EMA50 alignment (that is the late-trend worker's job)
- Entry: EMA12 reclaim (price crosses back above EMA12) + MACD hist expanding + RSI rising through 45
- Still requires daily regime FRIENDLY or MIXED + Risk ON
- Slightly tighter risk (1.25%) because entries are earlier and noisier

Independent of the late-trend worker. Own state file, own equity, own /status.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from flask import Flask, jsonify
import json, time, threading, os, warnings
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")

SYMBOL = "BTC-USD"
INTERVAL = "4h"
EMA_FAST, EMA_SLOW = 12, 26
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_LEN = 14
RSI_CROSS_LEVEL = 45
ATR_LEN, ATR_STOP_MULT = 14, 1.8
TRAIL_ACTIVATE_R, TRAIL_ATR_MULT = 0.7, 1.5
RISK_PCT = 0.0125
FEE_RT = 0.0004
INITIAL_EQUITY = 10_000.0
CHECK_INTERVAL = 300
DAILY_EMA_LEN = 50
MAX_DRAWDOWN_PCT = 0.08
MAX_CONSECUTIVE_LOSSES = 3
MIN_ATR_PCT = 0.004  # 0.4% of price floor to avoid weekend compression

STATE_FILE = Path("paper_state_reclaim.json")

def default_state():
    return {
        "worker": "reclaim",
        "philosophy": "early-trend / reclaim",
        "equity": INITIAL_EQUITY,
        "peak_equity": INITIAL_EQUITY,
        "position": 0,
        "entry_price": 0.0,
        "stop_price": 0.0,
        "orig_stop_pct": 0.0,
        "bars_held": 0,
        "last_bar_time": None,
        "trades": [],
        "equity_curve": [{"time": datetime.now(timezone.utc).isoformat(), "equity": INITIAL_EQUITY}],
        "last_update": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "regime": "unknown",
        "risk_state": "unknown",
        "agent_decision": "unknown",
        "last_price": None,
    }

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            for k, v in default_state().items():
                if k not in state:
                    state[k] = v
            return state
        except Exception:
            pass
    return default_state()

def save_state(state):
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def _norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df

def fetch_4h():
    df = yf.download(SYMBOL, period="60d", interval=INTERVAL, progress=False)
    if df.empty:
        return None
    df = _norm(df.dropna())
    df["ema12"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema26"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    macd = MACD(df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["rsi"] = RSIIndicator(df["close"], RSI_LEN).rsi()
    df["rsi_prev"] = df["rsi"].shift(1)
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], ATR_LEN).average_true_range()
    df["close_prev"] = df["close"].shift(1)
    df["ema12_prev"] = df["ema12"].shift(1)
    return df.dropna()

def fetch_daily():
    df = yf.download(SYMBOL, period="1y", interval="1d", progress=False)
    if df.empty:
        return None
    df = _norm(df.dropna())
    df["ema50"] = EMAIndicator(df["close"], DAILY_EMA_LEN).ema_indicator()
    return df.dropna()

def evaluate_regime():
    df = fetch_daily()
    if df is None or len(df) < DAILY_EMA_LEN + 5:
        return "unknown", {}
    last, prev = df.iloc[-1], df.iloc[-6] if len(df) >= 6 else df.iloc[0]
    price, ema50 = float(last["close"]), float(last["ema50"])
    slope_up = ema50 >= float(prev["ema50"])
    above = price > ema50
    if above and slope_up:
        regime = "FRIENDLY"
    elif (not above) or (not slope_up and price < ema50 * 0.98):
        regime = "HOSTILE"
    else:
        regime = "MIXED"
    return regime, {"daily_close": round(price, 2), "daily_ema50": round(ema50, 2),
                    "ema50_slope_up": slope_up, "price_above_ema50": above}

def evaluate_risk(state):
    equity = float(state.get("equity", INITIAL_EQUITY))
    peak = float(state.get("peak_equity", equity))
    if equity > peak:
        peak = equity
        state["peak_equity"] = peak
    drawdown = (peak - equity) / peak if peak > 0 else 0.0
    consecutive = 0
    for t in reversed(state.get("trades", [])):
        if t.get("pnl_pct", 0) < 0:
            consecutive += 1
        else:
            break
    risk_off = (drawdown >= MAX_DRAWDOWN_PCT) or (consecutive >= MAX_CONSECUTIVE_LOSSES)
    return ("RISK-OFF" if risk_off else "RISK-ON",
            {"equity": round(equity, 2), "peak_equity": round(peak, 2),
             "drawdown_pct": round(drawdown * 100, 2), "consecutive_losses": consecutive})

def get_signal(row):
    """Early-trend / reclaim: price reclaims EMA12 + hist expanding + RSI rising through 45."""
    # Reclaim: previously below or touching, now above EMA12
    reclaim = (float(row["close_prev"]) <= float(row["ema12_prev"])) and (float(row["close"]) > float(row["ema12"]))
    # Or simply price above EMA12 and EMA12 rising (softer reclaim)
    soft_reclaim = (float(row["close"]) > float(row["ema12"])) and (float(row["ema12"]) > float(row["ema12_prev"]))
    hist_expanding = (float(row["macd_hist"]) > float(row["macd_hist_prev"])) and (float(row["macd_hist"]) > 0)
    rsi_rising_through = (float(row["rsi_prev"]) < RSI_CROSS_LEVEL) and (float(row["rsi"]) >= RSI_CROSS_LEVEL)
    rsi_ok = float(row["rsi"]) >= RSI_CROSS_LEVEL and float(row["rsi"]) > float(row["rsi_prev"])
    return bool((reclaim or soft_reclaim) and hist_expanding and (rsi_rising_through or rsi_ok))

def run_once(state):
    df = fetch_4h()
    if df is None or len(df) < 5:
        state["status"] = "data_error"
        return state
    latest = df.iloc[-1]
    bar_time = str(latest.name)
    price = float(latest["close"])
    atr = float(latest["atr"])
    atr_pct = atr / price if price > 0 else 0

    regime, regime_detail = evaluate_regime()
    risk_state, risk_detail = evaluate_risk(state)
    state["regime"] = regime
    state["risk_state"] = risk_state
    state["regime_detail"] = regime_detail
    state["risk_detail"] = risk_detail
    allow = (regime in ("FRIENDLY", "MIXED")) and (risk_state == "RISK-ON")
    state["agent_decision"] = "CONTINUE" if allow else "PAUSE"

    if state.get("last_bar_time") == bar_time:
        state["status"] = "waiting_for_new_bar"
        state["last_price"] = price
        return state

    # Manage open position
    if state["position"] == 1:
        state["bars_held"] = state.get("bars_held", 0) + 1
        entry, stop = state["entry_price"], state["stop_price"]
        r_mult = (price - entry) / (entry - stop) if entry > stop else 0
        if r_mult >= TRAIL_ACTIVATE_R:
            new_stop = price - atr * TRAIL_ATR_MULT
            if new_stop > stop:
                state["stop_price"] = new_stop
        exit_now, reason = False, ""
        if price <= state["stop_price"]:
            exit_now, reason = True, "stop"
        elif float(latest["ema12"]) < float(latest["ema26"]):
            exit_now, reason = True, "trend_break"
        elif state["bars_held"] >= 60:
            exit_now, reason = True, "time"
        if exit_now:
            raw = (price - entry) / entry
            pnl = raw - FEE_RT
            risk_d = state["equity"] * RISK_PCT
            size = risk_d / state["orig_stop_pct"] if state.get("orig_stop_pct", 0) > 1e-8 else 0
            dollar = size * pnl
            state["equity"] += dollar
            if state["equity"] > state.get("peak_equity", state["equity"]):
                state["peak_equity"] = state["equity"]
            state.setdefault("trades", []).append({
                "time": bar_time, "side": "long",
                "entry": round(entry, 2), "exit": round(price, 2),
                "pnl_pct": round(pnl * 100, 3), "dollar": round(dollar, 2),
                "reason": reason, "bars": state["bars_held"],
                "equity_after": round(state["equity"], 2),
            })
            state["position"] = 0
            state["bars_held"] = 0
            risk_state, risk_detail = evaluate_risk(state)
            state["risk_state"], state["risk_detail"] = risk_state, risk_detail
            allow = (regime in ("FRIENDLY", "MIXED")) and (risk_state == "RISK-ON")
            state["agent_decision"] = "CONTINUE" if allow else "PAUSE"

    # New entry
    weekend = latest.name.weekday() >= 5 if hasattr(latest.name, "weekday") else False
    atr_ok = atr_pct >= MIN_ATR_PCT
    if state["position"] == 0 and allow and atr_ok and not weekend:
        if get_signal(latest) and atr > 0:
            stop_dist = max(atr * ATR_STOP_MULT, price * MIN_ATR_PCT * 1.5)
            state["position"] = 1
            state["entry_price"] = price
            state["stop_price"] = price - stop_dist
            state["orig_stop_pct"] = stop_dist / price
            state["bars_held"] = 0

    state["last_bar_time"] = bar_time
    state.setdefault("equity_curve", []).append({
        "time": datetime.now(timezone.utc).isoformat(), "equity": state["equity"]
    })
    if len(state["equity_curve"]) > 3000:
        state["equity_curve"] = state["equity_curve"][-2000:]
    state["status"] = "ok"
    state["last_price"] = price
    return state

def worker_loop():
    print("Reclaim worker loop started", flush=True)
    while True:
        try:
            state = load_state()
            state = run_once(state)
            save_state(state)
            pos = "LONG" if state["position"] == 1 else "FLAT"
            print(f"[reclaim {datetime.now().strftime('%H:%M:%S')}] {pos} | Eq ${state['equity']:.2f} | "
                  f"{state.get('regime')} / {state.get('risk_state')} / {state.get('agent_decision')}", flush=True)
        except Exception as e:
            print(f"Reclaim worker error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL)

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"service": "BTC Reclaim Worker (early-trend)", "endpoints": {"/status": "state", "/health": "health"}})

@app.route("/health")
def health():
    return jsonify({"status": "alive", "worker": "reclaim", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/status")
def status():
    return jsonify(load_state())

_started = False
_lock = threading.Lock()
def start_once():
    global _started
    with _lock:
        if not _started:
            threading.Thread(target=worker_loop, daemon=True).start()
            _started = True
start_once()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port)
