#!/usr/bin/env python3
"""
Targeted Model Training CLI
Allows training the XGBoost model with a custom performance target.
"""
import sys
import os
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from scripts.eval import MLBacktest
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Targeted Model Training')
    parser.add_argument('--days', type=str, default='90', help='Evaluation period in calendar days or "max"')
    parser.add_argument('--train-window', type=str, default='252', help='Training window in trading days or "max"')
    parser.add_argument('--target', type=float, default=0.85, help='Target combined score (Win Rate + W/L Ratio, 0.0 to 1.0)')
    parser.add_argument('--force', action='store_true', help='Replace champion if better. If False, saves with a new name.')
    parser.add_argument('--max-iter', type=int, default=5, help='Maximum optimization iterations')
    
    # Gen 5 Defaults
    parser.add_argument('--max-depth', type=int, default=6, help='XGBoost max tree depth (Gen 5 Default: 6)')
    parser.add_argument('--n-estimators', type=int, default=150, help='Number of boosting rounds (Gen 5 Default: 150)')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='XGBoost learning rate')
    
    # Trading Params (Defaults match Gen 4/Eval.py defaults)
    parser.add_argument('--stop-loss', type=float, default=0.05, help='Stop loss percentage (default: 0.05)')
    parser.add_argument('--take-profit', type=float, default=0.08, help='Take profit percentage (default: 0.08)')
    
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration (MPS on Mac, CUDA elsewhere)')

    args = parser.parse_args()
    
    # Update config with GPU setting
    if args.gpu:
        config.ml.use_gpu = True
    
    # Process 'max' arguments
    if args.days == 'max' or args.train_window == 'max':
        conn = sqlite3.connect(config.data.db_path)
        df_dates = pd.read_sql("SELECT MIN(date), MAX(date) FROM prices", conn)
        conn.close()
        
        db_min = pd.to_datetime(df_dates.iloc[0, 0])
        db_max = pd.to_datetime(df_dates.iloc[0, 1])
        total_days = (db_max - db_min).days
        
        if args.days == 'max':
            # Use 1 year if we have at least 1.5 years of history
            if total_days >= 365 * 1.5:
                eval_days = 365
            else:
                eval_days = max(30, total_days // 3)
            console.print(f"[dim]Auto-setting eval days to {eval_days} (max window)[/dim]")
        else:
            eval_days = int(args.days)
            
        if args.train_window == 'max':
            # Use at least 252 days if possible, or up to 3 years
            train_window = min(total_days - eval_days, 252 * 5)
            console.print(f"[dim]Auto-setting train window to {train_window} (max window)[/dim]")
        else:
            train_window = int(args.train_window)

    else:
        eval_days = int(args.days)
        train_window = int(args.train_window)

    console.print(f"[bold cyan]Starting Targeted Training for XGBOOST (Gen 5)[/bold cyan]")
    console.print(f"Target Score: [green]{args.target:.1%}[/green] | Max Iter: {args.max_iter}")
    console.print(f"Params: Depth={args.max_depth}, Est={args.n_estimators}, SL={args.stop_loss:.1%}, TP={args.take_profit:.1%}")

    # Phase 0: Data Prep
    backtester = MLBacktest()
    # Apply CLI args for SL/TP
    backtester.stop_loss_pct = args.stop_loss
    backtester.take_profit_pct = args.take_profit

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=eval_days)).strftime('%Y-%m-%d') # Using eval_days
    
    all_data = backtester._load_data(start_date, end_date, train_window=train_window) # Using train_window
    if all_data.empty:
        console.print("[red]No data available[/red]")
        return

    # Pre-calculating features
    ticker_groups = all_data.groupby('ticker')
    processed_data_list = []
    for ticker, group in ticker_groups:
        group = group.sort_values('date')
        group = backtester._add_features(group)
        processed_data_list.append(group)
    featured_data = pd.concat(processed_data_list).sort_values(['date', 'ticker'])

    # Session setup for caching
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(".cache", f"session_{session_timestamp}")
    if args.force:
        os.makedirs(session_dir, exist_ok=True)
        console.print(f"[dim]Session artifacts will be saved to: {session_dir}[/dim]")

    iteration = 0
    while iteration < args.max_iter:
        iteration += 1
        console.print(f"\n[bold magenta]Iteration {iteration}/{args.max_iter}[/bold magenta]")
        
        # Initialize backtester with retrain=True
        bt = MLBacktest(model_type='xgboost', retrain=True)
        
        # Apply Fixed Params (Stable Gen 5 approach)
        bt.stop_loss_pct = args.stop_loss
        bt.take_profit_pct = args.take_profit
        
        # Pass hyperparams down to config via temporary override logic if needed, 
        # but standard way is usually via config object. 
        # Assuming MLBacktest reads from config, we update config implicitly or assuming defaults are handled.
        # Actually, MLBacktest uses config.ml. Let's update `config.ml` dynamically?
        # NOTE: logic in eval.py uses 'config.ml'. config is imported from config.py.
        # We should update config.ml hyperparameters here if we want them to take effect.
        config.ml.xgboost.max_depth = args.max_depth
        config.ml.xgboost.n_estimators = args.n_estimators
        config.ml.xgboost.learning_rate = args.learning_rate
            
        results = bt.run(start_date=start_date, end_date=end_date, train_window=train_window, pre_loaded_data=featured_data)
        
        if results and 'win_rate' in results:
            monthly_wrs = [m['win_rate'] / 100.0 for m in results.get('monthly_metrics', [])]
            effective_wr = (sum(monthly_wrs) / len(monthly_wrs) * 0.7) + (min(monthly_wrs) * 0.3) if monthly_wrs else results['win_rate'] / 100.0
            
            # Calculate Win/Loss ratio for optimization
            avg_win = abs(results.get('avg_win', 0))
            avg_loss = abs(results.get('avg_loss', 1))
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Combined score: prioritize W/L ratio (60%) + win rate (40%)
            # Capping W/L benefit at 2.5x to prevent chasing outliers
            wl_ratio_normalized = min(wl_ratio / 2.5, 1.0) 
            combined_score = (effective_wr * 0.4) + (wl_ratio_normalized * 0.6)
            
            console.print(f"  Win Rate: [bold]{effective_wr:.1%}[/bold]")
            console.print(f"  Win/Loss Ratio: [bold]{wl_ratio:.2f}x[/bold]")
            console.print(f"  Combined Score: [bold cyan]{combined_score:.1%}[/bold cyan]")
            
            # Metadata for comparison
            import json
            metadata_path = 'models/champion_metadata.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    console.print(f"  [yellow]Warning: {metadata_path} is malformed. Resetting metadata.[/yellow]")
                    metadata = {'xgboost': {'win_rate': 0.0}}
            else:
                metadata = {'xgboost': {'win_rate': 0.0}}
                
            current_best_wr = metadata.get('xgboost', {}).get('win_rate', 0.0)
            
            if args.force:
                # Save every iteration to session folder
                cache_path = f"{session_dir}/iter_{iteration}.pkl"
                bt.global_xgb.save(cache_path)
                console.print(f"  [dim]Saved iteration {iteration} to {cache_path} (+.json)[/dim]")

                # Compare using combined score instead of just win rate
                current_best_score = metadata.get('xgboost', {}).get('combined_score', current_best_wr)
                
                # Check if we should update champion
                # If force is true, we update if it's better
                if combined_score > current_best_score:
                    console.print(f"  [green]Champion Updated! Score: {current_best_score:.1%} -> {combined_score:.1%}[/green]")
                    save_path = "models/global_xgb_champion.pkl"
                    bt.global_xgb.save(save_path)
                    
                    metadata['xgboost'] = {
                        'win_rate': float(effective_wr),
                        'wl_ratio': float(wl_ratio),
                        'combined_score': float(combined_score),
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'target_met': bool(combined_score >= args.target),
                        'hyperparams': {
                            'max_depth': int(args.max_depth),
                            'n_estimators': int(args.n_estimators),
                            'learning_rate': float(args.learning_rate),
                            'stop_loss': float(args.stop_loss),
                            'take_profit': float(args.take_profit)
                        }
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                else:
                    console.print(f"  [yellow]No improvement over current champion (score: {current_best_score:.1%}). Not replacing.[/yellow]")
            else:
                # Save with new name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                save_path = f"models/xgb_model_{timestamp}.pkl"
                bt.global_xgb.save(save_path)
                console.print(f"  [blue]Model saved to {save_path} (+.json)[/blue]")
            
            if combined_score >= args.target:
                console.print(f"[bold green]Target reached! Optimization complete.[/bold green]")
                console.print(f"  Final: WR={effective_wr:.1%}, W/L={wl_ratio:.2f}x, Score={combined_score:.1%}")
                break
        else:
            console.print("[red]Backtest iteration failed.[/red]")

if __name__ == "__main__":
    main()
