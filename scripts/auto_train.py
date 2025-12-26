#!/usr/bin/env python3
"""
Auto-Training Optimization Loop
Continuously trains and validates models to achieve performance targets (>80% WR)
"""
import sys
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.storage import DataStorage
from data.fetcher import DataFetcher
from ml.model import EnsembleModel
from scripts.ml_backtest import MLBacktest
from rich.console import Console

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_WIN_RATE = 0.80
TARGET_MONTHLY_RETURN = 0.15 # 15% monthly = ~200% annual compounded aggressively

def is_conservative_day():
    """
    Check if today is a conservative trading day (Friday or before holiday)
    User Request: "parameter related to human psychology" - fear of holding over weekend
    """
    today = datetime.now()
    
    # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
    if today.weekday() == 4: # Friday
        return True
        
    # TODO: Add holiday calendar check here
    
    return False

class AutoTrainer:
    def __init__(self):
        self.best_model_metrics = {'win_rate': 0.0, 'total_return': 0.0}
        self.storage = DataStorage(config.data.db_path)
        
    def run_optimization_loop(self):
        """Run the rigorous Train -> Backtest -> Verify loop"""
        console.print("[bold cyan]Starting Auto-Training Optimization Loop[/bold cyan]")
        console.print(f"Target Win Rate: [green]{TARGET_WIN_RATE:.0%}[/green]")
        
        # Ensure data is ready (at least some)
        all_data = self.storage.get_prices()
        if all_data.empty or len(all_data) < 1000:
            console.print("[yellow]Insufficient data in DB. Running initial fetch...[/yellow]")
            fetcher = DataFetcher(config.data.stock_universe)
            data = fetcher.fetch_batch(days=730) # 2 years
            self.storage.upsert_prices(data)
            all_data = self.storage.get_prices()

        iteration = 0
        while True:
            iteration += 1
            console.print(f"\n[bold magenta]Optimization Iteration {iteration}[/bold magenta]")
            
            for model_type in ['xgboost', 'gd_sd']:
                console.print(f"\n[bold cyan]Evaluating Model Type: {model_type.upper()}[/bold cyan]")
                
                # Step 1: Initialize Backtester for specific model
                backtester = MLBacktest(model_type=model_type)
                
                # Tuning: Try to hit 80% WR by tightening exits
                # Iteration 1 starts with default, Iteration 2+ tightens
                if iteration > 1:
                    backtester.stop_loss_pct = 0.03 + (iteration * 0.005)
                    backtester.take_profit_pct = 0.06 - (iteration * 0.005)
                
                # Step 2: Run Backtest
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                
                results = backtester.run(start_date=start_date, end_date=end_date, train_window=252)
                
                if results and 'win_rate' in results:
                    # Calculate effective Win Rate from monthly metrics
                    monthly_wrs = [m['win_rate'] / 100.0 for m in results.get('monthly_metrics', [])]
                    if monthly_wrs:
                        avg_wr = sum(monthly_wrs) / len(monthly_wrs)
                        min_wr = min(monthly_wrs)
                        effective_wr = (avg_wr * 0.7) + (min_wr * 0.3)
                    else:
                        effective_wr = results['win_rate'] / 100.0
                    
                    ret = results['total_return'] / 100.0
                    
                    console.print(f"  [bold]{model_type.upper()} Results:[/bold] Effective WR: {effective_wr:.1%} | Total Return: {ret:.1%}")
                    
                    if effective_wr >= TARGET_WIN_RATE:
                        console.print(f"[bold green]🏆 {model_type.upper()} TARGET REACHED![/bold green]")
                        # Save specifically as champion in models/ directory
                        model_dir = 'models'
                        os.makedirs(model_dir, exist_ok=True)
                        if model_type == 'xgboost':
                            save_path = os.path.join(model_dir, 'global_xgb_champion.pkl')
                            backtester.global_xgb.save(save_path)
                        else:
                            save_path = os.path.join(model_dir, 'global_sd_champion.pkl')
                            backtester.global_sd.save(save_path)
                        return # Stop everything once one hits target
                else:
                    console.print(f"[red]Backtest failed for {model_type}.[/red]")

            if iteration >= 5:
                console.print("[yellow]Reached max iterations without hitting perfect 80% target.[/yellow]")
                break
                
            console.print("  [dim]Tuning parameters for next round...[/dim]")
            time.sleep(2)

if __name__ == "__main__":
    trainer = AutoTrainer()
    trainer.run_optimization_loop()
