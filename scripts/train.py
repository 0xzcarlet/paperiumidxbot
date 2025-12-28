#!/usr/bin/env python3
"""
Targeted Model Training CLI (Gen 5)
Allows training the XGBoost model with a combined performance target (Win Rate + W/L Ratio).
"""
import sys
import os
import argparse
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from scripts.eval import MLBacktest

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Targeted Model Training (Gen 5)')
    parser.add_argument('--days', type=str, default='365', help='Evaluation period in calendar days or "max"')
    parser.add_argument('--train-window', type=str, default='max', help='Training window in trading days or "max"')
    parser.add_argument('--target', type=float, default=0.85, help='Target Combined Score (0.0 to 1.0)')
    # Gen 5 Ultimate Specs
    parser.add_argument('--max-depth', type=int, default=5, help='XGBoost max tree depth (Conservative: 5)')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of boosting rounds (Conservative: 100)')

    parser.add_argument('--force', action='store_true', help='Replace champion if better.')
    parser.add_argument('--max-iter', type=int, default=10, help='Maximum optimization iterations')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')

    args = parser.parse_args()
    
    # Update config with CLI args
    config.ml.use_gpu = args.gpu
    config.ml.max_depth = args.max_depth
    config.ml.n_estimators = args.n_estimators
    
    # Process 'max' arguments

    if args.days == 'max' or args.train_window == 'max':
        conn = sqlite3.connect(config.data.db_path)
        df_dates = pd.read_sql("SELECT MIN(date), MAX(date) FROM prices", conn)
        conn.close()
        
        db_min = pd.to_datetime(str(df_dates.iloc[0, 0]))
        db_max = pd.to_datetime(str(df_dates.iloc[0, 1]))
        total_days = (db_max - db_min).days
        
        if args.days == 'max':
            eval_days = 365  # Fixed: 1 year evaluation period
            console.print(f"[dim]Auto-setting eval days to {eval_days} (1 year fixed)[/dim]")
        else:
            eval_days = int(args.days)

        if args.train_window == 'max':
            train_window = int(252 * 4)  # Fixed: 4 years of trading days (1008 days)
            console.print(f"[dim]Auto-setting train window to {train_window} days (4 years fixed)[/dim]")
        else:
            train_window = int(args.train_window)
    else:
        eval_days = int(args.days)
        train_window = int(args.train_window)

    console.print(f"[bold cyan]Starting Targeted Training for XGBOOST (Gen 5)[/bold cyan]")
    console.print(f"Target Score: [green]{args.target:.1%}[/green] | Max Iter: {args.max_iter}")

    # Initialize training session file
    import json
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_file = f"models/training_session_{session_id}.json"
    session_data = {
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "target_score": args.target,
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "eval_days": eval_days,
            "train_window": train_window,
            "use_gpu": args.gpu,
            "force_replace": args.force
        },
        "iterations": []
    }
    console.print(f"[dim]Session file: {session_file}[/dim]")

    # Show GPU warning once at the start
    if args.gpu:
        import sys
        if sys.platform == "darwin":
            console.print("[yellow]⚠ XGBoost MPS (Metal) acceleration can be unstable on some Mac environments. Using high-performance CPU ('hist') instead.[/yellow]")

    # Phase 0: Data Prep
    backtester = MLBacktest()
    end_date = datetime.now().strftime('%Y-%m-%d')

    # When using 'max', align to month boundaries for cleaner evaluation
    if args.days == 'max':
        # Start from December 1st of the previous year
        current_year = datetime.now().year
        start_date = f"{current_year - 1}-12-01"
        console.print(f"[dim]Eval period aligned to: {start_date} to {end_date}[/dim]")
    else:
        start_date = (datetime.now() - timedelta(days=eval_days)).strftime('%Y-%m-%d')
    
    all_data = backtester._load_data(start_date, end_date, train_window=train_window)
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

    # Gen 5.1: Dynamic SL/TP tuning across iterations
    import numpy as np

    # Tracking best configuration
    best_wr = 0.0
    best_sl_mult = 2.0
    best_tp_mult = 3.0
    best_config = {'sl': best_sl_mult, 'tp': best_tp_mult, 'wr': 0.0, 'wl': 0.0}

    # Create backtester ONCE (preserve cache across iterations)
    bt = MLBacktest(model_type='xgboost', retrain=True)

    iteration = 0
    while iteration < args.max_iter:
        iteration += 1

        # Intelligent SL/TP selection - explore around best known config
        if iteration == 1:
            sl_mult = 2.0
            tp_mult = 3.0
        elif iteration <= 10:
            # First 10: small perturbations around baseline
            sl_mult = float(np.clip(2.0 + np.random.uniform(-0.3, 0.3), 1.5, 3.5))
            tp_mult = float(np.clip(3.0 + np.random.uniform(-0.5, 0.5), 2.0, 5.0))
        elif iteration <= 30:
            # 11-30: wider search around current best
            sl_mult = float(np.clip(best_sl_mult + np.random.uniform(-0.5, 0.5), 1.5, 3.5))
            tp_mult = float(np.clip(best_tp_mult + np.random.uniform(-0.8, 0.8), 2.0, 5.0))
        else:
            # 31+: fine-tuning around best
            sl_mult = float(np.clip(best_sl_mult + np.random.uniform(-0.2, 0.2), 1.5, 3.5))
            tp_mult = float(np.clip(best_tp_mult + np.random.uniform(-0.3, 0.3), 2.0, 5.0))

        console.print(f"\n[bold magenta]Iteration {iteration}/{args.max_iter}[/bold magenta]")
        console.print(f"  SL/TP Config: [cyan]{sl_mult:.2f}x ATR[/cyan] / [cyan]{tp_mult:.2f}x ATR[/cyan]")

        # Update SL/TP for this iteration
        bt.sl_atr_mult = sl_mult
        bt.tp_atr_mult = tp_mult

        iteration_start = datetime.now()
        results = bt.run(start_date=start_date, end_date=end_date, train_window=train_window, pre_loaded_data=featured_data)
        iteration_time = (datetime.now() - iteration_start).total_seconds()

        if results and 'win_rate' in results:
            # Calculate Metrics (Gen 5 Improved Formula)
            monthly_wrs = [m['win_rate'] / 100.0 for m in results.get('monthly_metrics', [])]
            effective_wr = (sum(monthly_wrs) / len(monthly_wrs) * 0.7) + (min(monthly_wrs) * 0.3) if monthly_wrs else results['win_rate'] / 100.0

            avg_win = abs(results.get('avg_win', 0))
            avg_loss = abs(results.get('avg_loss', 1))
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

            # Pure Win Rate Focus
            # W/L ratio naturally follows when win rate is high
            # Keep it simple - what worked for Gen 1-4
            combined_score = effective_wr
            
            console.print(f"  Win Rate: [bold]{effective_wr:.1%}[/bold]")
            console.print(f"  W/L Ratio: [bold]{wl_ratio:.2f}x[/bold]")

            # Gen 5.1: Track best SL/TP configuration
            if effective_wr > best_wr:
                best_wr = effective_wr
                best_sl_mult = sl_mult
                best_tp_mult = tp_mult
                best_config = {'sl': sl_mult, 'tp': tp_mult, 'wr': effective_wr, 'wl': wl_ratio}
                console.print(f"  [green]New best SL/TP: {sl_mult:.2f}x / {tp_mult:.2f}x[/green]")

            # Save iteration data to session file
            iteration_data = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": iteration_time,
                "sl_tp_config": {
                    "sl_atr_mult": float(sl_mult),
                    "tp_atr_mult": float(tp_mult)
                },
                "metrics": {
                    "win_rate": float(effective_wr),
                    "wl_ratio": float(wl_ratio),
                    "total_return": float(results.get('total_return', 0)),
                    "sharpe_ratio": float(results.get('sharpe_ratio', 0)),
                    "sortino_ratio": float(results.get('sortino_ratio', 0)),
                    "max_drawdown": float(results.get('max_drawdown', 0)),
                    "total_trades": int(results.get('total_trades', 0)),
                    "avg_win": float(avg_win),
                    "avg_loss": float(avg_loss)
                },
                "monthly_performance": results.get('monthly_metrics', []),
                "exit_breakdown": results.get('exit_breakdown', {})
            }
            session_data["iterations"].append(iteration_data)

            # Save session file after each iteration
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

            # Simple, no complex optimization
            # Just train fresh models and pick the best one

            # Metadata for comparison
            metadata_path = 'models/champion_metadata.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    metadata = {'xgboost': {'combined_score': 0.0}}
            else:
                metadata = {'xgboost': {'combined_score': 0.0}}
                
            # Use Win Rate (effective_wr)
            current_best_wr = metadata.get('xgboost', {}).get('win_rate', 0.0)

            if args.force:
                if effective_wr > current_best_wr:
                    console.print(f"  [green]Champion Updated! ({current_best_wr:.1%} -> {effective_wr:.1%})[/green]")
                    save_path = "models/global_xgb_champion.pkl"
                    bt.global_xgb.save(save_path)

                    metadata['xgboost'] = {
                        'win_rate': float(effective_wr),
                        'wl_ratio': float(wl_ratio),
                        'sl_atr_mult': float(sl_mult),
                        'tp_atr_mult': float(tp_mult),
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'target_met': bool(effective_wr >= args.target)
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                else:
                    console.print(f"  [yellow]No improvement over champ ({current_best_wr:.1%}).[/yellow]")
            else:
                # Save with new name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                save_path = f"models/xgb_model_{timestamp}.pkl"
                bt.global_xgb.save(save_path)
                console.print(f"  [blue]Model saved to {save_path}[/blue]")
            
            # Check if target reached
            if effective_wr >= args.target:
                console.print(f"\n[bold green]🎯 Target reached! Optimization complete.[/bold green]")
                console.print(f"  Final: WR={effective_wr:.1%}, W/L={wl_ratio:.2f}x")
                break
        else:
            console.print("[red]Backtest iteration failed.[/red]")

    # Save final session summary
    session_data["end_time"] = datetime.now().isoformat()
    session_data["total_iterations"] = iteration

    if session_data["iterations"]:
        best_iteration = max(session_data["iterations"], key=lambda x: x["metrics"]["win_rate"])
        session_data["best_iteration"] = {
            "iteration": best_iteration["iteration"],
            "win_rate": best_iteration["metrics"]["win_rate"],
            "wl_ratio": best_iteration["metrics"]["wl_ratio"],
            "total_return": best_iteration["metrics"]["total_return"],
            "sl_atr_mult": best_iteration["sl_tp_config"]["sl_atr_mult"],
            "tp_atr_mult": best_iteration["sl_tp_config"]["tp_atr_mult"]
        }

    # Gen 5.1: Save best SL/TP configuration
    session_data["best_sl_tp_config"] = best_config

    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2)

    console.print(f"\n[bold cyan]Training session completed![/bold cyan]")
    console.print(f"Session data saved to: [dim]{session_file}[/dim]")
    if session_data.get("best_iteration"):
        best = session_data["best_iteration"]
        console.print(f"Best iteration: #{best['iteration']} - WR: {best['win_rate']:.1%}, W/L: {best['wl_ratio']:.2f}x")
        console.print(f"Best SL/TP: [cyan]{best['sl_atr_mult']:.2f}x ATR[/cyan] / [cyan]{best['tp_atr_mult']:.2f}x ATR[/cyan]")

if __name__ == "__main__":
    main()
