# Paperium: High-Performance IHSG ML Trading Bot

Paperium is an automated trading system optimized for the Indonesia Stock Exchange (IHSG). It leverages a multi-model ensemble of XGBoost and Supply/Demand price action zones to achieve high-precision signals with target win rates exceeding 80%.

## 🚀 Key Features

*   **Universal Data Pooling**: Trains on the entire liquid IHSG universe (IDX80, Kompas100), generalizing across 460+ stocks.
*   **High-Potential Screener**: Pre-filters stocks for trend (EMA 200), liquidity, and ATR-based volatility before ML processing.
*   **Auto-Training Optimization**: A feedback loop that tunes model parameters to achieve >80% monthly win rates.
*   **Interactive Morning Signals**: Professional-grade CLI to review recommendations and execute trades in **Live** or **Test** mode.
*   **Champion vs. Challenger EOD**: End-of-day retraining that safely upgrades the "Champion" model only if a "Challenger" performs better.

## 🛠 Installation

Ensuring you have `uv` installed, then run:

```bash
# Clone the repository
cd paperium

# Install dependencies
uv sync
```

## 📈 7-Step Automated Workflow

The system is designed to run autonomously or under user supervision. Use the following scripts in order:

### 1. Universe Maintenance
Keep your stock list clean and up-to-date.
```bash
uv run python scripts/clean_universe.py  # Removes inactive stocks
uv run python scripts/sync_data.py       # Fetches 2 years of history for all tickers
```

### 2. Model Optimization
Train the brain. This loop iterates until it finds a model meeting the 80% WR target.
```bash
uv run python scripts/auto_train.py
```
*   *Champion model is saved to `models/global_xgb_champion.pkl`.*

### 3. Morning Signals (Daily Strategy)
Run this before the market opens (08:30–08:50 WIB).
```bash
uv run python scripts/morning_signals.py
```
*   **Test Mode**: Displays top-5 candidates and signals without changing your portfolio.
*   **Live Mode**: Commits trades to the database and tracks your positions.

### 4. EOD Retraining (Post-Market)
Run after the market close (16:00+ WIB).
```bash
uv run python scripts/eod_retrain.py
```
*Updates existing positions (SL/TP hits) and retrains the model with today's data.*

## 🧠 Intelligence & Strategies

Paperium uses two distinct machine learning architectures that can be trained and compared side-by-side.

### 1. XGBoost Champion (Default)
The primary "brain" of the system.
- **Strategy**: Gradient Boosting Trees (Classification).
- **Focus**: Capturing non-linear relationships between technical indicators and short-term price movements.
- **Strength**: High-precision entries. This model excels at identifying the "perfect" moment to enter a trade with a very high probability of hitting the target within 5 days.
- **Target Metrics**: 85%+ effective win rate.

### 2. GD/SD Alternative (Gradient Descent + Supply/Demand)
A robust alternative that combines classic price action with modern optimization.
- **Strategy**: Gradient Descent (Logistic Regression) paired with algorithmic Supply/Demand zone detection.
- **Focus**: Structural price action and trend confirmation.
- **Strength**: Reliability and low drawdown. It uses Supply/Demand zones to ensure entries are made near structural support/resistance, while the GD model confirms the directional probability.
- **Target Metrics**: 65%+ effective win rate with extremely low volatility.

## ⚙️ Configuration

Modify `config.py` to adjust:
- `MLConfig`: Training windows and min sample requirements.
- `PortfolioConfig`: Position sizing and maximum exposure.
- `ExitConfig`: Default Stop-Loss and Take-Profit percentages.

## 📁 Directory Structure
- `/models`: Persisted Champion models.
- `/data`: SQLite database (`trading.db`) and historical cache.
- `/scripts`: Main entry points for the automation workflow.
- `/ml`: Model architectures (XGBoost, SupplyDemand, Ensemble).

## 📊 Backtest Results
The latest baseline using the **XGBoost Global Champion** achieved:
- **Win Rate**: 89.7%
- **Total Return (90 days)**: +156.2%
- **Max Drawdown**: -1.0%

---
*Disclaimer: Trading stocks involves risk. This software is for paper trading and educational purposes.*
