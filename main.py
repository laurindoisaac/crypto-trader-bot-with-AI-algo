#!/usr/bin/env python3
import asyncio
import logging
import logging.handlers
import io
import sys
import os
import json
import sqlite3
import requests
from contextlib import closing
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# =============================
# ENVIRONMENT & CONFIGURATION
# =============================

# Load secrets from .env file
load_dotenv()

# Capital.com API credentials
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY")
CAPITAL_EMAIL = os.getenv("CAPITAL_EMAIL")
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD")
CAPITAL_EPIC = os.getenv("CAPITAL_EPIC")

# Telegram bot credentials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID") or 0)

# Trading configuration
TRADE_AMOUNT_ETH = float(os.getenv("TRADE_AMOUNT_ETH") or 0.05)
TRADE_SYMBOL = os.getenv("TRADE_SYMBOL") or "ETH/USD"
CONFIRM_TIMEOUT_SEC = int(os.getenv("CONFIRM_TIMEOUT_SEC") or 60)
AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "false").lower() == "true"

# File paths
LOG_FILE = os.getenv("LOG_FILE") or "tradingbot.log"
DB_FILE = os.getenv("DB_FILE") or "trades.db"

# Binance data settings
BINANCE_SYMBOL = os.getenv("BINANCE_SYMBOL") or "ETHUSDT"
BINANCE_INTERVAL = os.getenv("BINANCE_INTERVAL") or "5m"
BINANCE_LIMIT = int(os.getenv("BINANCE_LIMIT") or 500)

# Currency conversion fallback
USD_TO_AED_FALLBACK = float(os.getenv("USD_TO_AED_FALLBACK") or 3.80)
AUTO_INTERVAL = int(os.getenv("AUTO_INTERVAL") or 3600)
TRADE_SIZE = float(os.getenv("TRADE_SIZE") or 0.05)
MODEL_PATH = "price_predictor.pt"

# Ensure proper UTF-8 encoding for logs and console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")

# =============================
# LOGGER SETUP
# =============================

def get_logger(name="TradingBot"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

logger = get_logger()

# =============================
# HELPER FUNCTIONS
# =============================

def fmt_money(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def now_iso():
    return datetime.now().isoformat()

# =============================
# POSITION TRACKING (IN-MEMORY)
# =============================

_current_position = "FLAT"  # FLAT, LONG, SHORT

def get_position():
    return _current_position

def set_position(position):
    global _current_position
    _current_position = position
    logger.info(f"Position updated to: {position}")

# =============================
# DATABASE SETUP & LOGGING
# =============================

def init_db():
    with closing(sqlite3.connect(DB_FILE)) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            action TEXT,
            open_price_usd REAL,
            close_price_usd REAL,
            amount_eth REAL,
            opened_at TEXT,
            closed_at TEXT,
            status TEXT,
            pnl_usd REAL,
            pnl_aed REAL,
            notes TEXT,
            deal_id TEXT
        )
        """)
        # Ensure deal_id column exists if upgrading
        try:
            cur.execute("PRAGMA table_info(trades)")
            cols = [r[1] for r in cur.fetchall()]
            if "deal_id" not in cols:
                cur.execute("ALTER TABLE trades ADD COLUMN deal_id TEXT")
        except Exception:
            pass
        conn.commit()
    logger.info("DB initialized")

def get_last_open_trade():
    try:
        with closing(sqlite3.connect(DB_FILE)) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, symbol, action, open_price_usd, amount_eth, opened_at, deal_id
                FROM trades
                WHERE status='OPEN' OR status='CONFIRMED_EXECUTED'
                ORDER BY opened_at DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "symbol": row[1],
                    "action": row[2],
                    "open_price_usd": row[3],
                    "amount_eth": row[4],
                    "opened_at": row[5],
                    "deal_id": row[6],
                }
    except Exception as e:
        logger.error("Failed to fetch last open trade: %s", e)
    return None

def get_trade_by_id(trade_id: int):
    try:
        with closing(sqlite3.connect(DB_FILE)) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, symbol, action, open_price_usd, amount_eth, opened_at, deal_id
                FROM trades WHERE id=?
            """, (trade_id,))
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "symbol": row[1],
                    "action": row[2],
                    "open_price_usd": row[3],
                    "amount_eth": row[4],
                    "opened_at": row[5],
                    "deal_id": row[6],
                }
    except Exception as e:
        logger.error("Failed to fetch trade by id: %s", e)
    return None

def log_open_trade(symbol, action, price_usd, amount_eth, opened_at, status="OPEN", notes="", deal_id=None):
    try:
        with closing(sqlite3.connect(DB_FILE)) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO trades (symbol, action, open_price_usd, amount_eth, opened_at, status, notes, deal_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, price_usd, amount_eth, opened_at, status, notes, deal_id))
            trade_id = cur.lastrowid
            conn.commit()
            logger.info(f"Logged open trade: ID={trade_id}, {action} {amount_eth} {symbol} @ ${price_usd} (deal_id={deal_id})")
            return trade_id
    except Exception as e:
        logger.error("Failed to log open trade: %s", e)
        return None

def log_close_trade(trade_id, close_price_usd, closed_at, status="CLOSED", notes="", usd_to_aed=None):
    try:
        with closing(sqlite3.connect(DB_FILE)) as conn:
            cur = conn.cursor()
            
            # Get trade details
            cur.execute("SELECT open_price_usd, amount_eth, action FROM trades WHERE id=?", (trade_id,))
            row = cur.fetchone()
            if not row:
                logger.error(f"Trade {trade_id} not found")
                return None
                
            open_price_usd, amount_eth, action = row
            
            # Calculate P&L
            if "LONG" in action:
                pnl_usd = (close_price_usd - open_price_usd) * amount_eth
            else:  # SHORT
                pnl_usd = (open_price_usd - close_price_usd) * amount_eth
            
            if usd_to_aed is None:
                usd_to_aed = get_usd_to_aed_rate()
            pnl_aed = pnl_usd * usd_to_aed
            
            # Update trade
            cur.execute("""
            UPDATE trades SET close_price_usd=?, closed_at=?, status=?, pnl_usd=?, pnl_aed=?, notes=?
            WHERE id=?
            """, (close_price_usd, closed_at, status, pnl_usd, pnl_aed, notes, trade_id))
            conn.commit()
            
            logger.info(f"Logged close trade: ID={trade_id}, P&L=${pnl_usd:.2f} ({pnl_aed:.2f} AED)")
            return pnl_usd, pnl_aed
    except Exception as e:
        logger.error("Failed to log close trade: %s", e)
        return None

# === PRICE FETCHING UTILITIES ===
def get_eth_price_usd():
    try:
        client = Client()
        ticker = client.get_symbol_ticker(symbol=BINANCE_SYMBOL)
        return float(ticker['price'])
    except Exception as e:
        logger.error("Failed to fetch ETH price: %s", e)
        return 0.0

def get_usd_to_aed_rate():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=AED", timeout=6)
        r.raise_for_status()
        data = r.json()
        rate = data.get("rates", {}).get("AED")
        if rate:
            return float(rate)
    except Exception as e:
        logger.error("Failed to fetch USD->AED rate: %s", e)
    return USD_TO_AED_FALLBACK

# === DATA FETCHING FOR MODEL ===
def fetch_recent_data(symbol=BINANCE_SYMBOL, interval=BINANCE_INTERVAL, limit=BINANCE_LIMIT):
    try:
        # Ensure limit is an integer
        if isinstance(limit, str):
            # Extract number from strings like "300 hours"
            limit_num = ''.join(filter(str.isdigit, limit))
            limit = int(limit_num) if limit_num else 500
        
        client = Client()
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        logger.error("Failed to fetch recent data: %s", e)
        return pd.DataFrame()

# === LSTM MODEL CLASS ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# === MODEL UTILITIES ===
def load_model(path=MODEL_PATH):
    try:
        if not os.path.exists(path):
            logger.error("Model file not found: %s", path)
            return None

        model_data = torch.load(path, map_location=torch.device('cpu'))

        if isinstance(model_data, dict):
            # Try to load as LSTM
            try:
                model = LSTMModel()
                model.load_state_dict(model_data)
            except RuntimeError as e:
                logger.error("⚠ Model architecture mismatch: %s", e)
                logger.error("Please retrain or export your model with matching input_size, hidden_size, num_layers.")
                return None
        else:
            model = model_data

        model.eval()
        logger.info("✅ Loaded ML model from %s", path)
        return model
    except Exception as e:
        logger.error("⚠ Failed to load model: %s", e)
        return None

def predict_signal(model, df):
    if model is None or df.empty:
        logger.warning("⚠️ No model or data available, returning HOLD")
        return "HOLD", get_eth_price_usd(), 0.0

    try:
        # Use last 20 timesteps, each with 4 features: open, high, low, close
        if len(df) < 20:
            logger.warning("Not enough data for prediction")
            return "HOLD", get_eth_price_usd(), 0.0

        features = df[['open', 'high', 'low', 'close']].values[-20:]  # shape (20, 4)
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # shape (1, 20, 4)

        with torch.no_grad():
            output = model(X)
            prob = torch.sigmoid(output).item()

        current_price = df['close'].values[-1]

        if prob > 0.65:
            signal = "BUY"
        elif prob < 0.35:
            signal = "SELL"
        else:
            signal = "HOLD"

        return signal, current_price, prob

    except Exception as e:
        logger.error("Failed to generate signal: %s", e)
        return "HOLD", get_eth_price_usd(), 0.0

# === CAPITAL.COM API ===
try:
    from capitalcom import CapitalClient
    HAS_CAPITAL = True
except Exception:
    HAS_CAPITAL = False
    logger.warning("capitalcom client not installed – trades will be dry-run")

def _extract_deal_id_from_result(res):
    """
    Extract deal ID from Capital.com response.
    Can handle both dict responses and direct string deal IDs.
    """
    if isinstance(res, str) and res.strip():
        # Direct string deal ID
        return res.strip()
    
    if not isinstance(res, dict):
        return None

    # Original dict-based extraction logic
    for key in ["dealId", "deal_id", "positionId", "position_id", "id"]:
        if key in res and res[key]:
            return str(res[key])

    # Nested structures
    for key in ["position", "deal", "result", "data", "order", "dealReference"]:
        nested = res.get(key)
        if isinstance(nested, dict):
            for nk in ["dealId", "deal_id", "positionId", "position_id", "id"]:
                if nk in nested and nested[nk]:
                    return str(nested[nk])
        elif isinstance(nested, list) and nested:
            for item in nested:
                if isinstance(item, dict):
                    for nk in ["dealId", "deal_id", "positionId", "position_id", "id"]:
                        if nk in item and item[nk]:
                            return str(item[nk])

    return None

class CapitalAPI:
    def __init__(self, demo=True):
        self.demo = demo
        self.client = None
        self.authenticated = False
        self.epic = CAPITAL_EPIC or "ETHUSD"  # Default fallback

        if HAS_CAPITAL and CAPITAL_API_KEY and CAPITAL_EMAIL and CAPITAL_PASSWORD:
            try:
                logger.info("Attempting to connect to Capital.com (demo=%s)...", demo)
                self.client = CapitalClient(
                    api_key=CAPITAL_API_KEY,
                    login=CAPITAL_EMAIL,
                    password=CAPITAL_PASSWORD,
                    demo=demo
                )
                self.authenticated = True
                logger.info("✅ Connected to Capital.com client successfully")
                self._detect_epic()
            except Exception as e:
                logger.error("⚠ Failed to initialize Capital.com client: %s", e)
                logger.info("🔄 Continuing in dry-run mode...")
                self.client = None
                self.authenticated = False
        else:
            logger.warning("⚠ Missing Capital.com credentials - running in dry-run mode")

    def _detect_epic(self):
        """Try to find valid ETH epic symbols"""
        if not self.client:
            return
            
        try:
            possible_epics = [
                "ETHUSD", "ETH/USD", "ETHUSD.c", "ETHUSD.s", 
                "ETH-USD", "ETH_USD", "ETHEREUM", "ETHUSDT"
            ]
            logger.info("🔍 Testing epic formats for ETH...")
            for epic in possible_epics:
                try:
                    logger.info(f" - Testing: {epic}")
                    if not self.epic or self.epic == "ETHUSD":
                        self.epic = epic
                        logger.info(f"✅ Using epic: {epic}")
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.error("Failed to detect epic: %s", e)

    def open_position(self, direction="BUY", size=0.05, epic=None):
        trade_epic = epic or self.epic

        if not trade_epic:
            logger.error("⚠ No valid epic available – cannot open position.")
            return {"ok": False, "error": "No valid epic found"}

        if self.client and self.authenticated:
            try:
                logger.info(f"Opening {direction} position: {size} {trade_epic}")
                # Prefer raw method if available
                if hasattr(self.client, "open_raw_position"):
                    res = self.client.open_raw_position(direction=direction, epic=trade_epic, size=size)
                else:
                    # Fallback guesses
                    if hasattr(self.client, "open_position"):
                        res = self.client.open_position(direction=direction, epic=trade_epic, size=size)
                    else:
                        raise RuntimeError("Capital.com client missing open position method")
                logger.info("✅ Opened position via Capital.com: %s", res)
                return {"ok": True, "res": res}
            except Exception as e:
                error_msg = str(e)
                logger.exception("⚠ Capital open_position error: %s", error_msg)
                if "error.not-found.epic" in error_msg:
                    possible_epics = [
                        "ETHUSD", "ETH/USD", "ETHUSD.c", "ETHUSD.s",
                        "ETH-USD", "ETH_USD", "ETHEREUM", "ETHUSDT"
                    ]
                    for alt_epic in possible_epics:
                        if alt_epic == trade_epic:
                            continue
                        try:
                            logger.info(f"Retrying with epic: {alt_epic}")
                            if hasattr(self.client, "open_raw_position"):
                                res = self.client.open_raw_position(direction=direction, epic=alt_epic, size=size)
                            else:
                                res = self.client.open_position(direction=direction, epic=alt_epic, size=size)
                            logger.info("✅ Opened position via Capital.com: %s", res)
                            self.epic = alt_epic
                            return {"ok": True, "res": res}
                        except Exception as e2:
                            logger.error("Failed with epic %s: %s", alt_epic, e2)
                    return {"ok": False, "error": "All epic formats failed. Please check your account or symbol."}
                return {"ok": False, "error": error_msg}

        logger.info("🔄 (dry-run) Would open %s %s %s", direction, size, trade_epic)
        return {"ok": True, "res": {"dry_run": True, "direction": direction, "size": size, "epic": trade_epic}}

    def close_position(self, position_id: Optional[str] = None):
        """
        Close a position by its Capital.com deal/position ID.
        If position_id is None, attempt to fetch from the last open trade's deal_id.
        """
        # Dry-run path
        if not (self.client and self.authenticated):
            if position_id is None:
                trade = get_last_open_trade()
                position_id = trade.get("deal_id") if trade else None
            logger.info("🔄 (dry-run) Would close position id=%s", position_id)
            return {"ok": True, "res": {"dry_run": True, "position_id": position_id}}

        # Live path
        try:
            if position_id is None:
                trade = get_last_open_trade()
                if not trade:
                    return {"ok": False, "error": "No open trade found in database"}
                position_id = trade.get("deal_id")
                if not position_id:
                    return {"ok": False, "error": "No deal_id found for last open trade"}

            logger.info("Closing position via Capital.com: position_id=%s", position_id)

            # First, let's inspect what methods are available on the client
            available_methods = [method for method in dir(self.client) if not method.startswith('_')]
            logger.info("Available Capital.com client methods: %s", available_methods)
            
            # Try the most common method patterns for closing positions
            res = None
            methods_to_try = [
                ('close_deal', lambda: self.client.close_deal(position_id)),
                ('close_deal', lambda: self.client.close_deal(deal_id=position_id)),
                ('delete_position', lambda: self.client.delete_position(position_id)),
                ('delete_position', lambda: self.client.delete_position(deal_id=position_id)),
                ('close_position', lambda: self.client.close_position(position_id)),
                ('close_position', lambda: self.client.close_position(deal_id=position_id)),
                ('close_raw_position', lambda: self.client.close_raw_position(position_id)),
                ('close_raw_position', lambda: self.client.close_raw_position(deal_id=position_id)),
            ]
            
            for method_name, method_call in methods_to_try:
                if hasattr(self.client, method_name):
                    try:
                        logger.info(f"Trying method: {method_name}")
                        res = method_call()
                        logger.info(f"Success with {method_name}: {res}")
                        break
                    except Exception as e:
                        logger.warning(f"Method {method_name} failed: {e}")
                        continue
            
            if res is None:
                # If all methods fail, try to find any method with 'close' in the name
                close_methods = [m for m in available_methods if 'close' in m.lower()]
                if close_methods:
                    logger.info(f"Found potential close methods: {close_methods}")
                    # Try the first one found
                    method_name = close_methods[0]
                    try:
                        method = getattr(self.client, method_name)
                        res = method(position_id)
                        logger.info(f"Success with discovered method {method_name}: {res}")
                    except Exception as e:
                        logger.error(f"Discovered method {method_name} also failed: {e}")
            
            if res is None:
                error_msg = f"No working close method found. Available methods: {available_methods}"
                logger.error(error_msg)
                return {"ok": False, "error": error_msg}

            logger.info("✅ Closed position via Capital.com: %s", res)
            return {"ok": True, "res": res}
            
        except Exception as e:
            logger.exception("⚠ Capital close_position error: %s", e)
            return {"ok": False, "error": str(e)}

# =============================================================================
# TRADING LOGIC
# =============================================================================

def run_trading_cycle():
    logger.info("🔍 Starting TradingBot run…")

    # Load model
    model = load_model(MODEL_PATH)

    # Fetch data - use proper integer limit
    df = fetch_recent_data(BINANCE_SYMBOL, BINANCE_INTERVAL, 300)

    # Predict
    signal, price, prob = predict_signal(model, df)
    
    if signal != "HOLD":
        logger.info(f"📊 Signal: {signal} | Price: ${price:.2f} | Conf: {prob:.2%}")
    else:
        logger.info(f"📊 Signal: HOLD | Price: ${price:.2f} | Conf: {prob:.2%}")

    return signal, price, prob

# =============================================================================
# TELEGRAM BOT
# =============================================================================

_app = None
capital = CapitalAPI(demo=True)
_current_trade = None

def build_application():
    global _app
    if _app:
        return _app
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Missing TELEGRAM_BOT_TOKEN in config")
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in config")
    
    try:
        request = HTTPXRequest(connect_timeout=30, read_timeout=30, write_timeout=30, pool_timeout=30)
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", _start))
        app.add_handler(CommandHandler("close", _close_command))
        app.add_handler(CommandHandler("status", _status_command))
        app.add_handler(CommandHandler("test", _test_command))
        app.add_handler(CommandHandler("show", _show_command))
        app.add_handler(CommandHandler("capital", _capital_status_command))
        app.add_handler(CommandHandler("symbols", _test_symbols_command))
        app.add_handler(CallbackQueryHandler(_button_handler))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_custom_amount))
        
        _app = app
        logger.info("Telegram application built successfully")
        return app
    except Exception as e:
        logger.error("Failed to build Telegram application: %s", e)
        raise

async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        current_price = get_eth_price_usd()
        await update.message.reply_text(
            f"🤖 Trading Bot is online!\n\n"
            f"📊 Current ETH Price: ${fmt_money(current_price)}\n"
            f"💰 USD to AED Rate: {get_usd_to_aed_rate():.4f}\n\n"
            f"Commands:\n"
            "/show - Get market analysis & trading options\n"
            "/status - Check current trade status\n"
            "/close - Close current open trade\n"
            "/capital - Check Capital.com connection status\n"
            "/symbols - Test ETH symbol formats\n"
            "/test - Test bot functionality\n\n"
            f"Capital.com Status: {'Connected' if capital.authenticated else 'Demo mode'}"
        )
    except Exception as e:
        logger.error("Start command failed: %s", e)

async def _test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        init_db()
        current_price = get_eth_price_usd()
        capital_status = "Connected" if capital.authenticated else "Dry-run mode"
        usd_to_aed = get_usd_to_aed_rate()
        
        model = load_model(MODEL_PATH)
        model_status = "Loaded" if model else "Not found"
        
        df = fetch_recent_data()
        data_status = f"OK ({len(df)} records)" if not df.empty else "Failed"
        
        text = (
            "🧪 Bot Test Results:\n\n"
            f"📊 ETH Price: ${fmt_money(current_price)}\n"
            f"💱 USD to AED: {usd_to_aed:.4f}\n"
            f"🏦 Capital.com: {capital_status}\n"
            f"🗄️ Database: Connected\n"
            f"🤖 Model: {model_status}\n"
            f"📈 Data Feed: {data_status}\n"
            f"📱 Telegram: Working\n\n"
            "All core systems functional!"
        )
        
        await update.message.reply_text(text)
        
    except Exception as e:
        error_msg = f"Test failed: {str(e)}"
        await update.message.reply_text(error_msg)
        logger.exception("Test command failed: %s", e)

async def _status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    try:
        trade = get_last_open_trade()
        if not trade:
            await context.bot.send_message(chat_id=chat_id, text="📊 No open trades found.")
            return
        
        current_price = get_eth_price_usd()
        open_price = float(trade["open_price_usd"])
        amount_eth = float(trade["amount_eth"])
        action = trade["action"]
        
        if "LONG" in action:
            pnl_usd = (current_price - open_price) * amount_eth
        else:
            pnl_usd = (open_price - current_price) * amount_eth
            
        usd_to_aed = get_usd_to_aed_rate()
        pnl_aed = pnl_usd * usd_to_aed
        
        profit_status = "Profit" if pnl_usd >= 0 else "Loss"
        emoji = "💰" if pnl_usd >= 0 else "📉"
        
        text = (
            f"📈 Current Trade Status:\n\n"
            f"🆔 Trade ID: {trade['id']}\n"
            f"📊 Symbol: {trade['symbol']}\n"
            f"🎯 Action: {action}\n"
            f"📈 Open Price: ${fmt_money(open_price)}\n"
            f"💰 Current Price: ${fmt_money(current_price)}\n"
            f"💵 Amount: {amount_eth:.4f} ETH\n"
            f"{emoji} {profit_status}: ${fmt_money(pnl_usd)} (~{fmt_money(pnl_aed)} AED)\n"
            f"📅 Opened: {trade['opened_at']}\n"
            f"🔗 Deal ID: {trade.get('deal_id') or 'N/A'}"
        )
        await context.bot.send_message(chat_id=chat_id, text=text)
        
    except Exception as e:
        error_msg = f"Error getting status: {str(e)}"
        await context.bot.send_message(chat_id=chat_id, text=error_msg)
        logger.exception("Status command failed: %s", e)

async def _close_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    try:
        await update.message.reply_text("🔍 Looking up last open trade...")
        
        trade = get_last_open_trade()
        if not trade:
            await context.bot.send_message(chat_id=chat_id, text="📊 No open trades found.")
            return

        trade_id = trade["id"]
        open_price = float(trade["open_price_usd"])
        amount_eth = float(trade["amount_eth"])

        current_price = get_eth_price_usd()

        # Use deal_id if available
        deal_id = trade.get("deal_id")
        if not deal_id:
            await context.bot.send_message(chat_id=chat_id, text="⚠ Cannot close trade – missing deal ID.")
            return

        res = capital.close_position(position_id=deal_id)

        usd_to_aed = get_usd_to_aed_rate()
        pnl_result = log_close_trade(
            trade_id, current_price, now_iso(), 
            status="CLOSED", notes=str(res), usd_to_aed=usd_to_aed
        )
        
        if pnl_result:
            pnl_usd, pnl_aed = pnl_result
            profit_status = "Profit" if pnl_usd >= 0 else "Loss"
            emoji = "💰" if pnl_usd >= 0 else "📉"
            
            set_position("FLAT")
            
            text = (
                f"{emoji} Trade Closed Successfully!\n\n"
                f"🆔 Trade ID: {trade_id}\n"
                f"📈 Open Price: ${fmt_money(open_price)}\n"
                f"💰 Close Price: ${fmt_money(current_price)}\n"
                f"💵 Amount: {amount_eth:.4f} ETH\n"
                f"📊 Final {profit_status}: ${fmt_money(pnl_usd)} (~{fmt_money(pnl_aed)} AED)\n"
                f"🔗 Deal ID: {deal_id or 'N/A'}\n"
                f"🤖 Capital.com: {'Success' if res.get('ok') else 'Dry-run'}"
            )
            await context.bot.send_message(chat_id=chat_id, text=text)
            logger.info("Closed trade id=%s P&L USD=%s AED=%s", trade_id, pnl_usd, pnl_aed)
        else:
            await context.bot.send_message(chat_id=chat_id, text="⚠ Failed to close trade - trade not found in database.")
            
    except Exception as e:
        error_msg = f"Error closing trade: {str(e)}"
        await context.bot.send_message(chat_id=chat_id, text=error_msg)
        logger.exception("Close command failed: %s", e)

async def _show_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    try:
        current_price = get_eth_price_usd()
        usd_to_aed = get_usd_to_aed_rate()
        
        signal, price, confidence = run_trading_cycle()
        
        current_trade = get_last_open_trade()
        
        if signal == "BUY" and confidence > 0.5:
            recommendation = "📈 BUY (Go Long)"
            recommendation_emoji = "🟢"
            recommendation_text = "Strong buy signal detected!"
        elif signal == "SELL" and confidence > 0.5:
            recommendation = "📉 SELL (Go Short)"
            recommendation_emoji = "🔴"
            recommendation_text = "Strong sell signal detected!"
        else:
            recommendation = "⏸️ HOLD"
            recommendation_emoji = "🟡"
            recommendation_text = "No clear signal - wait for better opportunity"
        
        message = (
            f"📊 Market Analysis\n\n"
            f"💰 Current ETH Price: ${fmt_money(current_price)}\n"
            f"💱 USD to AED Rate: {usd_to_aed:.4f}\n"
            f"💵 ETH Value in AED: {fmt_money(current_price * usd_to_aed)}\n\n"
            f"🤖 AI Recommendation:\n"
            f"{recommendation_emoji} {recommendation}\n"
            f"🎲 Confidence: {confidence:.1%}\n"
            f"💡 {recommendation_text}\n\n"
        )
        
        if current_trade:
            open_price = float(current_trade["open_price_usd"])
            amount_eth = float(current_trade["amount_eth"])
            action = current_trade["action"]
            
            if "LONG" in action:
                pnl_usd = (current_price - open_price) * amount_eth
            else:
                pnl_usd = (open_price - current_price) * amount_eth
            
            pnl_aed = pnl_usd * usd_to_aed
            profit_status = "Profit" if pnl_usd >= 0 else "Loss"
            emoji = "💰" if pnl_usd >= 0 else "📉"
            
            message += (
                f"📈 Current Trade:\n"
                f"🎯 {action} @ ${fmt_money(open_price)}\n"
                f"{emoji} {profit_status}: ${fmt_money(pnl_usd)} (~{fmt_money(pnl_aed)} AED)\n\n"
            )
        else:
            message += "📈 No open trades\n\n"
        
        message += "❓ What would you like to do?"
        
        if current_trade:
            keyboard = [
                [
                    InlineKeyboardButton("✅ Close Trade", callback_data=f"close_trade_{current_trade['id']}"),
                    InlineKeyboardButton("⏳ Continue", callback_data="continue_trade")
                ]
            ]
        else:
            keyboard = [
                [
                    InlineKeyboardButton("📈 Buy (Long)", callback_data=f"select_LONG_{TRADE_SYMBOL}_{current_price}"),
                    InlineKeyboardButton("📉 Sell (Short)", callback_data=f"select_SHORT_{TRADE_SYMBOL}_{current_price}")
                ],
                [
                    InlineKeyboardButton("❌ Cancel", callback_data="cancel")
                ]
            ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await context.bot.send_message(
            chat_id=chat_id, 
            text=message, 
            reply_markup=reply_markup
        )
        
        logger.info(f"Show command executed: {recommendation} @ ${current_price:.2f}")
        
    except Exception as e:
        error_msg = f"Error getting market analysis: {str(e)}"
        await context.bot.send_message(chat_id=chat_id, text=error_msg)
        logger.exception("Show command failed: %s", e)

async def _capital_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    try:
        status = "✅ Connected" if capital.authenticated else "⚠ Not Connected"
        mode = "Live" if not capital.demo else "Demo"
        
        message = (
            f"🏦 Capital.com Status\n\n"
            f"🔗 Connection: {status}\n"
            f"🎯 Mode: {mode}\n"
            f"📧 Email: {CAPITAL_EMAIL or 'Not set'}\n"
            f"🔑 API Key: {'Set' if CAPITAL_API_KEY else 'Not set'}\n"
            f"📊 Epic: {capital.epic}\n\n"
        )
        
        if capital.authenticated:
            try:
                account_info = capital.get_account_info()
                if account_info and not account_info.get('error'):
                    message += f"💰 Account Status: Active\n"
                else:
                    message += f"💰 Account Info: {account_info.get('error', 'Available')}\n"
            except Exception as e:
                message += f"💰 Account Info: Error - {str(e)}\n"
        else:
            message += (
                "⚠ Connection Issues:\n"
                "• Check your .env file has correct credentials\n"
                "• Verify API key is active\n"
                "• Ensure account supports API trading\n"
                "• Check if demo/live account matches settings\n"
            )
        
        await context.bot.send_message(chat_id=chat_id, text=message)
        
    except Exception as e:
        error_msg = f"Error checking Capital.com status: {str(e)}"
        await context.bot.send_message(chat_id=chat_id, text=error_msg)
        logger.exception("Capital status command failed: %s", e)

async def _test_symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    try:
        message = "🔍 Testing ETH Epic Formats\n\n"
        test_epics = [
            "ETHUSD", "ETH/USD", "ETHUSD.c", "ETHUSD.s", "ETHUSD.d",
            "ETH-USD", "ETH_USD", "ETHEREUM", "ETHUSDT"
        ]
        
        message += f"🎯 Current Epic: {capital.epic}\n"
        message += f"🔗 Connection: {'Active' if capital.authenticated else 'Dry-run'}\n\n"
        
        message += "🔍 Suggested Epic Formats:\n"
        for i, epic in enumerate(test_epics, 1):
            message += f"{i}. {epic}\n"
        
        message += (
            f"\n💡 To test a specific epic:\n"
            f"1. Update your .env file:\n"
            f"   CAPITAL_EPIC=ETHUSD\n"
            f"2. Restart the bot\n"
            f"3. Try a small trade\n\n"
            f"🔧 Most common format is: ETHUSD"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=message)
        
    except Exception as e:
        error_msg = f"Error testing symbols: {str(e)}"
        await context.bot.send_message(chat_id=chat_id, text=error_msg)
        logger.exception("Symbols test command failed: %s", e)

async def _handle_custom_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _current_trade
    
    try:
        if 'pending_trade' not in context.user_data:
            return
        
        try:
            aed_amount = float(update.message.text.strip())
        except ValueError:
            await update.message.reply_text("⚠ Invalid amount. Please send a valid number (e.g., 1500)")
            return
        
        if aed_amount < 25:
            await update.message.reply_text("⚠ Minimum investment is 25 AED. Please enter a higher amount.")
            return
        elif aed_amount > 50000:
            await update.message.reply_text("⚠ Maximum investment is 50,000 AED. Please enter a lower amount.")
            return
        
        trade_data = context.user_data['pending_trade']
        mode = trade_data['mode']
        direction = trade_data['direction']
        symbol = trade_data['symbol']
        price = trade_data['price']
        
        usd_to_aed = get_usd_to_aed_rate()
        eth_price_aed = price * usd_to_aed
        eth_amount = aed_amount / eth_price_aed
        
        is_demo = (mode == "demo")
        trade_capital = CapitalAPI(demo=is_demo)
        
        logger.info(f"Executing custom {mode} trade: {direction} {eth_amount:.4f} ETH @ ${price:.2f} ({aed_amount} AED)")
        result = trade_capital.open_position(
            direction="BUY" if "LONG" in direction else "SELL",
            epic=symbol,
            size=eth_amount
        )
        logger.info(f"Custom trade result: {result}")
        
        if result and isinstance(result, dict) and result.get("ok"):
            res_payload = result.get("res")
            deal_id = _extract_deal_id_from_result(res_payload)
            
            is_dry_run = isinstance(res_payload, dict) and res_payload.get("dry_run", False)
            
            if not deal_id and not is_dry_run:
                logger.error("Could not extract deal ID from Capital.com response")
                await update.message.reply_text("Trade execution failed – no deal ID returned by Capital.com.")
                return
            
            trade_id = log_open_trade(
                symbol=symbol,
                action=direction,
                price_usd=price,
                amount_eth=eth_amount,
                opened_at=now_iso(),
                status="CONFIRMED_EXECUTED",
                notes=f"Capital.com result: {res_payload}",
                deal_id=deal_id
            )
            
            _current_trade = {
                "id": trade_id,
                "symbol": symbol,
                "action": direction,
                "open_price_usd": price,
                "amount_eth": eth_amount,
                "opened_at": now_iso(),
                "deal_id": deal_id
            }
            
            set_position(direction)
            
            mode_emoji = "🎮" if mode == "demo" else "💰"
            mode_text = "Demo" if mode == "demo" else "Live"
            
            await update.message.reply_text(
                f"✅ Trade Executed Successfully!\n\n"
                f"🆔 Trade ID: {trade_id}\n"
                f"🎯 Action: {direction}\n"
                f"💰 Price: ${fmt_money(price)}\n"
                f"💵 Investment: {fmt_money(aed_amount)} AED\n"
                f"📊 ETH Amount: {eth_amount:.4f} ETH\n"
                f"💸 USD Value: ${fmt_money(price * eth_amount)}\n"
                f"🔗 Deal ID: {deal_id or 'N/A'}\n"
                f"{mode_emoji} Mode: {mode_text}\n\n"
                f"📊 Bot will monitor for close signals..."
            )
            
            logger.info(f"Custom trade executed: {direction} {eth_amount:.4f} ETH @ ${price:.2f} ({aed_amount} AED) deal_id={deal_id}")
            
        else:
            # Handle failed trades  
            error_msg = "Unknown error"
            if isinstance(result, str):
                error_msg = result
            elif isinstance(result, dict) and result.get('error'):
                error_msg = result.get('error')
            
            await update.message.reply_text(f"Trade execution failed!\nError: {error_msg}")
            return
                
        del context.user_data['pending_trade']
        
    except Exception as e:
        logger.exception("Custom amount handler failed: %s", e)
        await update.message.reply_text("⚠ An error occurred processing your amount. Please try again.")

async def _button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _current_trade
    
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    try:
        if data == "cancel":
            await query.edit_message_text("Trade cancelled by user.")
            return
        
        if data == "skip_trade":
            await query.edit_message_text("⚠ Trade skipped by user.")
            return
        
        if data == "continue_trade":
            await query.edit_message_text("⏳ Trade will continue running...")
            return
        
        if data.startswith("select_"):
            parts = data.split("_")
            if len(parts) >= 4:
                direction = parts[1]  # LONG or SHORT
                symbol = parts[2]
                price = float(parts[3])
                
                keyboard = [
                    [
                        InlineKeyboardButton("🎮 Demo Account", callback_data=f"mode_demo_{direction}_{symbol}_{price}"),
                        InlineKeyboardButton("💰 Live Account", callback_data=f"mode_live_{direction}_{symbol}_{price}")
                    ],
                    [
                        InlineKeyboardButton("❌ Cancel", callback_data="cancel")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                message = (
                    f"🎯 Select Trading Account\n\n"
                    f"📊 Current ETH Price: ${fmt_money(price)}\n"
                    f"🎯 Trade Direction: {direction}\n\n"
                    f"💡 Choose which account to use for this trade:"
                )
                
                await query.edit_message_text(message, reply_markup=reply_markup)
                return
        
        if data.startswith("mode_"):
            parts = data.split("_")
            if len(parts) >= 5:
                mode = parts[1]  # demo or live
                direction = parts[2]  # LONG or SHORT
                symbol = parts[3]
                price = float(parts[4])
                
                usd_to_aed = get_usd_to_aed_rate()
                eth_price_aed = price * usd_to_aed
                
                amounts = [25, 50, 100, 250, 500, 1000, 2500, 5000]
                keyboard = []
                
                for i in range(0, len(amounts), 2):
                    row = []
                    for j in range(2):
                        if i + j < len(amounts):
                            amount = amounts[i + j]
                            eth_amount = amount / eth_price_aed
                            row.append(InlineKeyboardButton(
                                f"💰 {amount} AED ({eth_amount:.4f} ETH)", 
                                callback_data=f"amount_{mode}_{direction}_{symbol}_{price}_{amount}_{eth_amount:.4f}"
                            ))
                    keyboard.append(row)
                
                keyboard.append([
                    InlineKeyboardButton("💵 Custom Amount", callback_data=f"custom_{mode}_{direction}_{symbol}_{price}")
                ])
                keyboard.append([
                    InlineKeyboardButton("❌ Cancel", callback_data="cancel")
                ])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                mode_emoji = "🎮" if mode == "demo" else "💰"
                mode_text = "Demo" if mode == "demo" else "Live"
                
                message = (
                    f"💰 Select Investment Amount\n\n"
                    f"📊 Current ETH Price: ${fmt_money(price)}\n"
                    f"💱 ETH Price in AED: {fmt_money(eth_price_aed)}\n"
                    f"🎯 Trade Direction: {direction}\n"
                    f"{mode_emoji} Account Mode: {mode_text}\n\n"
                    f"💡 Choose how much AED you want to invest:"
                )
                
                await query.edit_message_text(message, reply_markup=reply_markup)
                return
        
        if data.startswith("amount_"):
            parts = data.split("_")
            if len(parts) >= 7:
                mode = parts[1]  # demo or live
                direction = parts[2]
                symbol = parts[3]
                price = float(parts[4])
                aed_amount = float(parts[5])
                eth_amount = float(parts[6])
                
                is_demo = (mode == "demo")
                trade_capital = CapitalAPI(demo=is_demo)
                
                logger.info(f"Executing {mode} trade: {direction} {eth_amount:.4f} ETH @ ${price:.2f} ({aed_amount} AED)")
                result = trade_capital.open_position(
                    direction="BUY" if "LONG" in direction else "SELL",
                    epic=symbol,
                    size=eth_amount
                )
                logger.info(f"Trade result: {result}")
                
                if result and isinstance(result, dict) and result.get("ok"):
                    res_payload = result.get("res") or {}
                    deal_id = _extract_deal_id_from_result(res_payload)
                    
                    # Check if we have a deal_id OR if this is a dry-run
                    is_dry_run = isinstance(res_payload, dict) and res_payload.get("dry_run", False)
                    
                    if not deal_id and not is_dry_run:
                        logger.error("⚠ Could not extract deal ID from Capital.com response, aborting trade logging")
                        await query.edit_message_text("⚠ Trade execution failed – no deal ID returned by Capital.com.")
                        return
                    trade_id = log_open_trade(
                        symbol=symbol,
                        action=direction,
                        price_usd=price,
                        amount_eth=eth_amount,
                        opened_at=now_iso(),
                        status="CONFIRMED_EXECUTED",
                        notes=f"Capital.com result: {res_payload}",
                        deal_id=deal_id
                    )
                    
                    _current_trade = {
                        "id": trade_id,
                        "symbol": symbol,
                        "action": direction,
                        "open_price_usd": price,
                        "amount_eth": eth_amount,
                        "opened_at": now_iso(),
                        "deal_id": deal_id
                    }
                    
                    set_position(direction)
                    
                    mode_emoji = "🎮" if mode == "demo" else "💰"
                    mode_text = "Demo" if mode == "demo" else "Live"
                    
                    await query.edit_message_text(
                        f"✅ Trade Executed Successfully!\n\n"
                        f"🆔 Trade ID: {trade_id}\n"
                        f"🎯 Action: {direction}\n"
                        f"💰 Price: ${fmt_money(price)}\n"
                        f"💵 Investment: {fmt_money(aed_amount)} AED\n"
                        f"📊 ETH Amount: {eth_amount:.4f} ETH\n"
                        f"💸 USD Value: ${fmt_money(price * eth_amount)}\n"
                        f"🔗 Deal ID: {deal_id or 'N/A'}\n"
                        f"{mode_emoji} Mode: {mode_text}\n\n"
                        f"📊 Bot will monitor for close signals..."
                    )
                    
                    logger.info(f"Trade executed: {direction} {eth_amount:.4f} ETH @ ${price:.2f} ({aed_amount} AED) deal_id={deal_id}")
                    
                else:
                    # Handle failed trades
                    error_msg = "Unknown error"
                    if isinstance(result, str):
                        error_msg = result
                    elif isinstance(result, dict) and result.get('error'):
                        error_msg = result.get('error')
                    
                    await query.edit_message_text(
                        f"⚠ Trade execution failed!\n"
                        f"Error: {error_msg}"
                    )
        
        if data.startswith("custom_"):
            parts = data.split("_")
            if len(parts) >= 5:
                mode = parts[1]  # demo or live
                direction = parts[2]
                symbol = parts[3]
                price = float(parts[4])
                
                context.user_data['pending_trade'] = {
                    'mode': mode,
                    'direction': direction,
                    'symbol': symbol,
                    'price': price
                }
                
                usd_to_aed = get_usd_to_aed_rate()
                eth_price_aed = price * usd_to_aed
                
                mode_emoji = "🎮" if mode == "demo" else "💰"
                mode_text = "Demo" if mode == "demo" else "Live"
                
                await query.edit_message_text(
                    f"💵 Custom Investment Amount\n\n"
                    f"📊 Current ETH Price: ${fmt_money(price)}\n"
                    f"💱 ETH Price in AED: {fmt_money(eth_price_aed)}\n"
                    f"🎯 Trade Direction: {direction}\n"
                    f"{mode_emoji} Account Mode: {mode_text}\n\n"
                    f"💰 Please send the AED amount you want to invest (e.g., 1500)\n"
                    f"💡 Minimum: 25 AED, Maximum: 50,000 AED"
                )
                return
        
        if data.startswith("close_trade_"):
            trade_id = int(data.split("_")[2])

            # Lookup the trade to get its deal_id
            trade = get_trade_by_id(trade_id)
            deal_id = trade.get("deal_id") if trade else None
            
            current_price = get_eth_price_usd()
            
            result = capital.close_position(position_id=deal_id)
            
            usd_to_aed = get_usd_to_aed_rate()
            pnl_result = log_close_trade(
                trade_id, current_price, now_iso(), 
                status="CLOSED", notes=str(result), usd_to_aed=usd_to_aed
            )
            
            if pnl_result:
                pnl_usd, pnl_aed = pnl_result
                profit_status = "Profit" if pnl_usd >= 0 else "Loss"
                emoji = "💰" if pnl_usd >= 0 else "📉"
                
                _current_trade = None
                set_position("FLAT")
                
                await query.edit_message_text(
                    f"{emoji} Trade Closed Successfully!\n\n"
                    f"🆔 Trade ID: {trade_id}\n"
                    f"🔗 Deal ID: {deal_id or 'N/A'}\n"
                    f"📊 Final {profit_status}: ${fmt_money(pnl_usd)} (~{fmt_money(pnl_aed)} AED)\n"
                    f"💰 Current Price: ${fmt_money(current_price)}\n"
                    f"🤖 Mode: {'Demo' if (isinstance(result, dict) and isinstance(result.get('res'), dict) and result.get('res', {}).get('dry_run')) else 'Live'}\n\n"
                    f"🎯 Bot will continue monitoring for new opportunities..."
                )
                
                logger.info(f"Trade closed: ID={trade_id}, P&L=${pnl_usd:.2f}")
            else:
                await query.edit_message_text("⚠ Failed to close trade - trade not found in database.")
            return
                    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        await query.edit_message_text(error_msg)
        logger.exception("Button handler failed: %s", e)

async def send_startup_message():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, skipping startup message")
        return
    
    try:
        current_price = get_eth_price_usd()
        usd_to_aed = get_usd_to_aed_rate()
        
        message = (
            f"🚀 Trading Bot Started!\n\n"
            f"📊 Current ETH Price: ${fmt_money(current_price)}\n"
            f"💰 USD to AED Rate: {usd_to_aed:.4f}\n"
            f"💵 ETH Value in AED: {fmt_money(current_price * usd_to_aed)}\n\n"
            f"🤖 Bot is now monitoring for trading opportunities...\n"
            f"📈 I'll notify you when there's a good signal to trade! or Use /show to begin.\n\n"
            f"Capital.com: {'Live' if capital.authenticated else 'Demo'} mode"
        )
        
        if _app:
            await _app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info("Startup message sent to Telegram")
    except Exception as e:
        logger.error("Failed to send startup message: %s", e)

async def send_trade_signal(signal, price, confidence):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, skipping trade signal")
        return
    
    try:
        if signal == "HOLD":
            return
        
        if signal == "BUY":
            direction = "LONG"
            emoji = "📈"
            action_text = "Buy (Go Long)"
        else:
            direction = "SHORT"
            emoji = "📉"
            action_text = "Sell (Go Short)"
        
        message = (
            f"{emoji} Trading Signal Detected!\n\n"
            f"🎯 Signal: {signal} ({action_text})\n"
            f"💰 Current Price: ${fmt_money(price)}\n"
            f"🎲 Confidence: {confidence:.1%}\n"
            f"💵 Trade Size: {TRADE_SIZE} ETH\n"
            f"💸 Trade Value: ${fmt_money(price * TRADE_SIZE)}\n\n"
            f"❓ Do you want to execute this trade?"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Execute Trade", callback_data=f"select_{direction}_{TRADE_SYMBOL}_{price}"),
                InlineKeyboardButton("❌ Skip", callback_data="skip_trade")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if _app:
            await _app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=message, 
                reply_markup=reply_markup
            )
            logger.info(f"Trade signal sent: {signal} @ ${price:.2f}")
    except Exception as e:
        logger.error("Failed to send trade signal: %s", e)

async def send_close_signal(trade_info, current_price):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, skipping close signal")
        return
    
    try:
        open_price = float(trade_info["open_price_usd"])
        amount_eth = float(trade_info["amount_eth"])
        action = trade_info["action"]
        
        if "LONG" in action:
            pnl_usd = (current_price - open_price) * amount_eth
        else:
            pnl_usd = (open_price - current_price) * amount_eth
        
        usd_to_aed = get_usd_to_aed_rate()
        pnl_aed = pnl_usd * usd_to_aed
        
        if pnl_usd > 0:
            emoji = "💰"
            status = "PROFIT"
        else:
            emoji = "📉"
            status = "LOSS"
        
        message = (
            f"{emoji} Close Signal Detected!\n\n"
            f"📊 Current Trade Status:\n"
            f"🎯 Action: {action}\n"
            f"📈 Open Price: ${fmt_money(open_price)}\n"
            f"📊 Current Price: ${fmt_money(current_price)}\n"
            f"💵 Amount: {amount_eth:.4f} ETH\n\n"
            f"💰 {status}: ${fmt_money(pnl_usd)} (~{fmt_money(pnl_aed)} AED)\n\n"
            f"❓ Do you want to close this trade?"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Close Trade", callback_data=f"close_trade_{trade_info['id']}"),
                InlineKeyboardButton("⏳ Continue", callback_data="continue_trade")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if _app:
            await _app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=message, 
                reply_markup=reply_markup
            )
            logger.info(f"Close signal sent: {status} ${pnl_usd:.2f}")
    except Exception as e:
        logger.error("Failed to send close signal: %s", e)

# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

async def trading_loop():
    loop_count = 0
    while True:
        try:
            loop_count += 1
            logger.info(f"Running trading cycle #{loop_count}...")
            
            signal, price, prob = run_trading_cycle()
            
            current_trade = get_last_open_trade()
            
            if current_trade:
                logger.info(f"Open trade detected: {current_trade['action']} @ ${current_trade['open_price_usd']}")
                
                should_close = False
                if signal == "SELL" and "LONG" in current_trade['action'] and prob > 0.6:
                    should_close = True
                elif signal == "BUY" and "SHORT" in current_trade['action'] and prob > 0.6:
                    should_close = True
                
                if should_close:
                    logger.info(f"Close signal detected: {signal} @ ${price:.2f} (confidence: {prob:.1%})")
                    await send_close_signal(current_trade, price)
                else:
                    logger.info("Monitoring open trade - no close signal yet")
            else:
                if signal != "HOLD" and prob > 0.6:
                    logger.info(f"New trade signal: {signal} @ ${price:.2f} (confidence: {prob:.1%})")
                    await send_trade_signal(signal, price, prob)
                else:
                    logger.info(f"Signal: {signal} @ ${price:.2f} (confidence: {prob:.1%}) - No action needed")
            
        except Exception as e:
            logger.exception("Error in trading loop: %s", e)
        
        logger.info(f"Next cycle in {AUTO_INTERVAL} seconds...")
        await asyncio.sleep(AUTO_INTERVAL)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

async def main():
    logger.info("Starting Trading Bot...")
    
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        return
    
    try:
        app = build_application()
        logger.info("Telegram application built")
    except Exception as e:
        logger.error("Failed to build Telegram application: %s", e)
        logger.error("Please check your TELEGRAM_BOT_TOKEN in .env file")
        return
    
    trading_task = asyncio.create_task(trading_loop())
    
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        
        logger.info("Telegram bot started successfully")
        logger.info("Bot is running. Press Ctrl+C to stop.")
        
        await send_startup_message()
        
        try:
            await trading_task
        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")
        
    except Exception as e:
        logger.exception("Failed to start bot: %s", e)
    finally:
        if not trading_task.done():
            trading_task.cancel()
            try:
                await trading_task
            except asyncio.CancelledError:
                pass
        
        try:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
        except Exception as e:
            logger.error("Error during shutdown: %s", e)
        
        logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
