# Paperium: High-Performance IHSG Day Trading Bot

Paperium is an automated **AI-driven Day Trading system** optimized for the Indonesia Stock Exchange (IHSG). It uses a sophisticated "Dual-Brain" architecture (XGBoost + Structural GD/SD) to identify high-probability intraday setups.

Designed for both beginners and advanced traders, Paperium focuses on **high win rates (>80%)** by combining machine learning with structural price action analysis.

---

## Quick Start Guide

If you're new, the easiest way to operate Paperium is via the **Unified Runner**:

```bash
# Start the interactive dashboard
python run.py
```

This menu-driven interface will guide you through:
1.  **Morning Ritual**: Generating signals before the market opens.
2.  **Evening Update**: Updating results and retraining for tomorrow.
3.  **Model Lab**: Training and optimizing your AI models.

---

## Detailed Workflow

For those who want full control, here is the complete step-by-step process of running your trading floor.

### 1. Initial Setup & Data Prep
Before training any AI, you need clean data.
```bash
# 1. Clean the stock list (removes illiquid/suspended stocks)
uv run python scripts/clean_universe.py

# 2. Fetch historical data (fills your database with years of price action)
uv run python scripts/sync_data.py
```

### 2. Training your "Brain"
You have two ways to prepare your models. The goal is to create a **"Champion"** (your best-performing model).

*   **Option A: Auto-Training (Recommended)**
    Continuously loops through parameters until it finds a model that hits the >80% win rate target.
    ```bash
    uv run python scripts/auto_train.py --days 90
    ```
*   **Option B: Targeted Training**
    Focus on a specific model type with a custom performance target.
    ```bash
    uv run python scripts/train_model.py --type xgboost --target 0.85 --days 180
    ```

### 3. Evaluating & Backtesting
Never trade blindly. Use the backtester to see how your "Champion" would have performed on fresh data or different timeframes.

*   **Simple Backtest**: Use the latest 3 months to verify accuracy.
    ```bash
    uv run python scripts/ml_backtest.py --start 2024-06-01 --end 2024-09-30
    ```
*   **Custom Windows**: Test how much history the model needs to "learn" effectively.
    ```bash
    # Test using a 500-day learning window
    uv run python scripts/ml_backtest.py --window 500
    ```

### 4. Getting Morning Signals (Day Trading)
Run this daily between **08:30 – 08:50 WIB** before the Jakarta market opens.
```bash
uv run python scripts/morning_signals.py
```
*   **Mode 1 (Test)**: Just see the recommendations (Scan only).
*   **Mode 2 (Live)**: Execute and track the trades in your virtual portfolio.

### 5. Daily Management & Retraining (EOD)
Run this after the market closes (**16:00+ WIB**) to close out your day trades and learn from today's market.
```bash
uv run python scripts/eod_retrain.py
```
*   **Settlement**: Automatically calculates PnL for positions that hit Stop-Loss or Take-Profit.
*   **Warm Start**: Retrains the models using a "Warm Start"—building on existing intelligence rather than starting from scratch.

---

## Core Philosophy: The Dual-Brain

Paperium doesn't rely on just one indicator. It uses two distinct logic sets:
1.  **The Predictor (XGBoost)**: A machine learning model that looks at dozens of technical features to predict the probability of a positive return tomorrow.
2.  **The Architect (GD/SD)**: Uses Gradient Descent to find structural Support and Demand zones. It only buys when price is at a high-conviction structural floor.

---

## Configuration Hints

All primary settings are held in `config.py`. Key areas for beginners:
- **Risk Management**: Change `max_loss_pct` (Stop Loss) and `min_profit_pct` (Take Profit) in `ExitConfig`.
- **Position Sizing**: Adjust `max_positions` and `base_position_pct` in `PortfolioConfig`.
- **Learning**: Change the default `training_window` in `MLConfig`.

---

## Key Directories
- `/models`: Stores your `.pkl` Champion models (Incremental learning).
- `/data`: Your SQLite database (`ihsg_trading.db`) containing price history and portfolio state.
- `/scripts`: The executable tools for your daily workflow.

---
*Disclaimer: Trading stocks involves significant risk. This bot is a tool for decision support. Always use Test Mode before committing to live trading.*
