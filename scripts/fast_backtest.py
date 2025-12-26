#!/usr/bin/env python3
"""
Fast Backtest Script
Optimized backtesting with pre-calculated signals

Usage:
    uv run python scripts/fast_backtest.py --start 2024-01-01 --end 2025-09-30
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import config
from data.storage import DataStorage
from data.fetcher import get_sector_mapping

logging.basicConfig(level=logging.WARNING)
console = Console()


class FastBacktest:
    """
    Optimized backtester using vectorized operations.
    Pre-calculates all signals before simulation.
    """
    
    def __init__(self):
        self.storage = DataStorage(config.data.db_path)
        self.sector_mapping = get_sector_mapping()
        
        # Trading parameters - Best performing config
        self.initial_capital = 100_000_000  # 100M IDR
        self.max_positions = 10
        self.buy_fee = 0.0015
        self.sell_fee = 0.0025
        self.stop_loss_pct = 0.05  # 5%
        self.take_profit_pct = 0.08  # 8%
        self.max_hold_days = 5
    
    def run(self, start_date: str, end_date: str):
        """Run fast backtest."""
        console.print(Panel.fit(
            f"[bold green]IHSG Fast Backtest[/bold green]\n"
            f"[dim]{start_date} to {end_date}[/dim]",
            border_style="green"
        ))
        
        # Load and prepare data
        console.print("\n[yellow]📊 Loading and preparing data...[/yellow]")
        all_data = self._load_data(start_date, end_date)
        
        if all_data.empty:
            console.print("[red]No data available[/red]")
            return
        
        # Pre-calculate all signals
        console.print("[yellow]🔍 Pre-calculating signals...[/yellow]")
        signals_df = self._calculate_all_signals(all_data)
        
        console.print(f"  ✓ {len(signals_df)} signal records generated")
        
        # Run simulation
        console.print("[yellow]🎯 Running simulation...[/yellow]")
        results = self._simulate(signals_df, start_date, end_date)
        
        # Display results
        self._display_results(results)
        
        return results
    
    def _load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data from database."""
        # Load with some buffer for indicator calculation
        buffer_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
        
        data = self.storage.get_prices(start_date=buffer_start, end_date=end_date)
        
        if not data.empty:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values(['ticker', 'date'])
            console.print(f"  ✓ Loaded {len(data)} records for {data['ticker'].nunique()} stocks")
        
        return data
    
    def _calculate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate signals for all stocks using vectorized operations."""
        results = []
        
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            if len(ticker_data) < 50:
                continue
            
            # Calculate indicators
            ticker_data = self._add_indicators(ticker_data)
            
            # Calculate composite score
            ticker_data['score'] = self._calculate_score(ticker_data)
            
            # Add sector
            ticker_data['sector'] = self.sector_mapping.get(ticker, 'Other')
            
            results.append(ticker_data)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators efficiently."""
        # Returns
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volatility
        df['volatility'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['rel_volume'] = df['volume'] / df['volume_sma']
        
        # Z-score (mean reversion)
        df['zscore'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()
        
        return df
    
    def _calculate_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite trading score."""
        score = pd.Series(0.0, index=df.index)
        
        # Momentum (25%)
        momentum = df['return_5d'].clip(-0.2, 0.2) * 5  # Normalized
        score += 0.25 * momentum.fillna(0)
        
        # Mean reversion (25%)
        zscore_signal = -df['zscore'].clip(-3, 3) / 3  # Inverted: low z = buy
        score += 0.25 * zscore_signal.fillna(0)
        
        # RSI (20%)
        rsi_signal = (50 - df['rsi']) / 50  # Below 50 = bullish
        score += 0.20 * rsi_signal.fillna(0)
        
        # MACD (15%)
        macd_norm = df['macd_hist'] / (df['close'] * 0.02 + 1)
        macd_signal = macd_norm.clip(-1, 1)
        score += 0.15 * macd_signal.fillna(0)
        
        # Trend (15%)
        trend = np.where(df['close'] > df['sma_20'], 0.5, -0.5)
        trend = np.where(df['sma_20'] > df['sma_50'], trend + 0.5, trend - 0.5)
        score += 0.15 * pd.Series(trend / 2, index=df.index)
        
        return score.clip(-1, 1)
    
    def _simulate(self, signals_df: pd.DataFrame, start_date: str, end_date: str) -> Dict:
        """Run trading simulation."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Filter to simulation period
        sim_data = signals_df[(signals_df['date'] >= start) & (signals_df['date'] <= end)]
        trading_dates = sorted(sim_data['date'].unique())
        
        console.print(f"  Simulating {len(trading_dates)} trading days...")
        
        # Initialize
        cash = self.initial_capital
        positions = {}  # ticker -> {shares, entry_price, entry_date, stop_loss, take_profit}
        trades = []
        equity_curve = []
        
        from rich.progress import Progress
        
        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Simulating...", total=len(trading_dates))
            
            for i, date in enumerate(trading_dates):
                day_data = sim_data[sim_data['date'] == date]
                
                # Update positions and check exits
                for ticker in list(positions.keys()):
                    ticker_day = day_data[day_data['ticker'] == ticker]
                    if ticker_day.empty:
                        continue
                    
                    row = ticker_day.iloc[0]
                    pos = positions[ticker]
                    
                    # Check stop loss
                    if row['low'] <= pos['stop_loss']:
                        exit_price = pos['stop_loss']
                        proceeds = pos['shares'] * exit_price * (1 - self.sell_fee)
                        cash += proceeds
                        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                        trades.append({
                            'ticker': ticker, 'entry_date': pos['entry_date'],
                            'exit_date': date, 'entry_price': pos['entry_price'],
                            'exit_price': exit_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'STOP_LOSS'
                        })
                        del positions[ticker]
                        continue
                    
                    # Check take profit
                    if row['high'] >= pos['take_profit']:
                        exit_price = pos['take_profit']
                        proceeds = pos['shares'] * exit_price * (1 - self.sell_fee)
                        cash += proceeds
                        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                        trades.append({
                            'ticker': ticker, 'entry_date': pos['entry_date'],
                            'exit_date': date, 'entry_price': pos['entry_price'],
                            'exit_price': exit_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        del positions[ticker]
                        continue
                    
                    # Check time stop
                    days_held = (date - pos['entry_date']).days
                    if days_held >= self.max_hold_days:
                        exit_price = row['close']
                        proceeds = pos['shares'] * exit_price * (1 - self.sell_fee)
                        cash += proceeds
                        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                        trades.append({
                            'ticker': ticker, 'entry_date': pos['entry_date'],
                            'exit_date': date, 'entry_price': pos['entry_price'],
                            'exit_price': exit_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TIME_STOP'
                        })
                        del positions[ticker]
                
                # Rebalance daily when we have open slots
                if len(positions) < self.max_positions:
                    # Get top candidates for today (lowered threshold to 0.15)
                    candidates = day_data[
                        (day_data['score'] >= 0.15) & 
                        (~day_data['ticker'].isin(positions.keys()))
                    ].nlargest(self.max_positions - len(positions), 'score')
                    
                    for _, cand in candidates.iterrows():
                        if len(positions) >= self.max_positions:
                            break
                        
                        ticker = cand['ticker']
                        entry_price = cand['close']
                        atr_pct = cand['atr_pct'] if pd.notna(cand['atr_pct']) else 0.02
                        
                        # Position size: equal weight
                        position_value = min(cash * 0.12, self.initial_capital * 0.1)
                        shares = int(position_value / entry_price)
                        
                        if shares <= 0:
                            continue
                        
                        cost = shares * entry_price * (1 + self.buy_fee)
                        if cost > cash * 0.95:
                            continue
                        
                        cash -= cost
                        positions[ticker] = {
                            'shares': shares,
                            'entry_price': entry_price,
                            'entry_date': date,
                            'stop_loss': entry_price * (1 - max(self.stop_loss_pct, atr_pct * 2)),
                            'take_profit': entry_price * (1 + max(self.take_profit_pct, atr_pct * 3))
                        }
                
                # Calculate equity
                equity = cash
                for ticker, pos in positions.items():
                    ticker_day = day_data[day_data['ticker'] == ticker]
                    if not ticker_day.empty:
                        equity += pos['shares'] * ticker_day.iloc[0]['close']
                
                equity_curve.append({'date': date, 'equity': equity})
                progress.update(task, advance=1)
        
        # Close remaining positions
        if positions and len(trading_dates) > 0:
            final_date = trading_dates[-1]
            final_data = sim_data[sim_data['date'] == final_date]
            
            for ticker, pos in positions.items():
                ticker_day = final_data[final_data['ticker'] == ticker]
                if ticker_day.empty:
                    continue
                
                exit_price = ticker_day.iloc[0]['close']
                pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                trades.append({
                    'ticker': ticker, 'entry_date': pos['entry_date'],
                    'exit_date': final_date, 'entry_price': pos['entry_price'],
                    'exit_price': exit_price, 'pnl_pct': pnl_pct,
                    'exit_reason': 'END_OF_BACKTEST'
                })
        
        return self._calculate_metrics(trades, equity_curve)
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        if equity_df.empty:
            return {}
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Daily returns
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        returns = equity_df['daily_return'].dropna()
        
        # Sharpe ratio (5% risk-free)
        rf = 0.05 / 252
        sharpe = np.sqrt(252) * (returns.mean() - rf) / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio
        downside = returns[returns < 0]
        sortino = np.sqrt(252) * (returns.mean() - rf) / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
        
        # Max drawdown
        running_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - running_max) / running_max
        max_dd = drawdown.min()
        
        # Trade statistics
        if not trades_df.empty:
            winners = trades_df[trades_df['pnl_pct'] > 0]
            losers = trades_df[trades_df['pnl_pct'] <= 0]
            win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
            avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
        else:
            win_rate = avg_win = avg_loss = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'total_trades': len(trades_df),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades_df,
            'equity_curve': equity_df
        }
    
    def _display_results(self, results: Dict):
        """Display backtest results."""
        if not results:
            console.print("[red]No results[/red]")
            return
        
        console.print("\n" + "=" * 60)
        console.print("[bold]📊 BACKTEST RESULTS[/bold]")
        console.print("=" * 60)
        
        # Main metrics
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        ret_style = "green" if results['total_return'] > 0 else "red"
        
        table.add_row("Initial Capital", f"Rp {results['initial_capital']:,.0f}")
        table.add_row("Final Equity", f"Rp {results['final_equity']:,.0f}")
        table.add_row("Total Return", f"[{ret_style}]{results['total_return']:.1f}%[/{ret_style}]")
        table.add_row("", "")
        table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        table.add_row("Sortino Ratio", f"{results['sortino_ratio']:.2f}")
        table.add_row("Max Drawdown", f"[red]{results['max_drawdown']:.1f}%[/red]")
        table.add_row("", "")
        table.add_row("Total Trades", f"{results['total_trades']}")
        table.add_row("Win Rate", f"{results['win_rate']:.1f}%")
        table.add_row("Avg Win", f"[green]+{results['avg_win']:.1f}%[/green]")
        table.add_row("Avg Loss", f"[red]{results['avg_loss']:.1f}%[/red]")
        
        console.print(table)
        
        # Exit breakdown
        if not results['trades'].empty:
            console.print("\n[bold]Exit Breakdown[/bold]")
            exit_counts = results['trades']['exit_reason'].value_counts()
            for reason, count in exit_counts.items():
                console.print(f"  {reason}: {count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Backtest')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-09-30', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    bt = FastBacktest()
    bt.run(args.start, args.end)


if __name__ == "__main__":
    main()
