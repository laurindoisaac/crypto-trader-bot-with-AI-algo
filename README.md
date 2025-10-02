# Crypto Trader Bot with AI algo (ver. 1.05)
In 2017, I started trading, thinking I would become the next Warren Buffett. I made hundreds in profit but then gave most of it back when the bear market hit. Clearly, I was not cut out for trading!
I can now turn my trading ideas into working code in just a few minutes, backtest them to see if they actually work, and confidently launch them for live trading if I like the results.

This Crypto Trader Bot integrates machine learning price prediction, trading API (demo/live), and a Telegram bot interface with inline controls.

## ✅ Features
- **ML Model (LSTM)**: Predicts ETH/SOL/BTC price direction using Binance OHLC data.
- **Live/Demo Trading**: Supports both live and demo modes.
- **Telegram Bot UI**: Inline buttons for opening/closing trades with confirmation.
- **Position Tracking**: Keeps track of open/closed trades in SQLite.
- **EURO Conversion**: All trade values shown in **EUR** with real-time conversion.
- **Custom Investment**: Choose preset or custom EUR investment amounts.
- **Safe Execution**: Bot only trades on explicit Telegram confirmation.

## 🛠 Tech Stack
- **Language**: Python 3
- **ML Framework**: PyTorch
- **Data Feed**: Binance API
- **Messaging**: Telegram Bot API
- **Database**: SQLite
- **Libraries**:
  - `torch`, `numpy`, `pandas`
  - `python-telegram-bot`
  - `binance`
  - `python-dotenv`
  - `requests`

## ▶️ Usage
- `/show` → Run AI prediction, get trading recommendation, choose BUY/SELL.
- `/status` → Show current open trade details and PnL.
- `/close` → Close the last open trade.
- `/capital` → Show connection status.
- `/symbols` → Suggest valid epic formats.

## ▶️ Trading_flags:
  
- underlying_instrument: "NIFTY 50"
- chart_timeframe: "5minute"
- product_type: "MIS" # or "NRML"
- order_variety: "REGULAR"
- risk_per_trade_percent: 1.0 # e.g., 1% of capital
- stop_loss_percent: 15.0 # 15% stop-loss on the option premium
- max_trades_per_day: 5
- paper_trading: true # Set to false for live trading
- enable_gemini_loss_analysis: true
- enable_natural_language_prompt: true
- strategy_reassessment_period_minutes: 60
- use_rag: false
- rag_min_trading_days: 5

## ▶️ Architecture Overview
The application is built on a modular, agent-based architecture designed for scalability and resilience.

- Orchestrator (trading_bot.py): The central brain of the application. It manages the main event loop, state transitions (e.g., AWAITING_SIGNAL, IN_POSITION), and coordinates all other agents.
- Agents (agents.py):
- OrderExecutionAgent: Handles all aspects of order placement, sizing, and communication with the Kite API. Implements the "Isolated Worker Pattern" to ensure thread-safe order execution.
- PositionManagementAgent: Manages active trades, applying stop-loss, trailing stop-loss, and other risk management rules.

## ▶️ Architecture OverviewIntelligence Layer:

- langgraph_agent.py: Interfaces with the Gemini LLM to select strategies.
- sentiment_agent.py: Fetches and analyzes news to determine market sentiment.
- market_context.py: Identifies current market conditions (VIX, IV, etc.).
- rag_service.py: The RAG engine that retrieves historical performance from logs to augment the AI's prompts.

## ▶️ Strategy & Indicators:

- strategy_factory.py: A library of all trading strategies.
- indicator_calculator.py & indicators.py: Calculate all necessary technical indicators.

## ▶️ Reporting & Persistence:

- reporting.py: Manages the generation and emailing of performance reports.
- output/: Directory where trade logs and backtest results are stored.

Disclaimer:
This software is provided for educational and experimental purposes only. Algorithmic trading involves substantial risk and is not suitable for all investors. The authors and contributors are not responsible for any financial losses incurred through the use of this software. Always test thoroughly in paper trading mode before deploying with real capital.
