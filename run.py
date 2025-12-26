#!/usr/bin/env python3
"""
Paperium Unified Runner
Provides an easy-to-use CLI to run all project workflows.
"""
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    clear_screen()
    console.print(Panel.fit(
        "[bold cyan]Paperium Trading System[/bold cyan]\n"
        "[dim]XGBoost Intelligence[/dim]",
        border_style="cyan"
    ))
    
    console.print("\n[bold yellow]Main Menu:[/bold yellow]")
    console.print("1. [bold green]Morning Ritual[/bold green] (Generate Signals)")
    console.print("2. [bold magenta]Evening Update[/bold magenta] (EOD Retrain & Evaluation)")
    console.print("3. [bold cyan]Model Lab[/bold cyan] (Targeted Model Training)")
    console.print("4. [bold white]Full Strategy Sweep[/bold white] (Global Auto-Train)")
    console.print("5. [bold blue]Backtest & Verification[/bold blue] (Evaluate Models)")
    console.print("0. Exit")
    
    choice = Prompt.ask("\nSelect action", choices=["1", "2", "3", "4", "5", "0"], default="1")
    
    if choice == "1":
        subprocess.run(["uv", "run", "python", "scripts/morning_signals.py"])
    elif choice == "2":
        subprocess.run(["uv", "run", "python", "scripts/eod_retrain.py"])
    elif choice == "3":
        train_menu()
    elif choice == "4":
        subprocess.run(["uv", "run", "python", "scripts/auto_train.py"])
    elif choice == "5":
        backtest_menu()
    elif choice == "0":
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)
        
    input("\nPress Enter to return to menu...")
    main_menu()

def backtest_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Backtest & Verification Lab[/bold cyan]", border_style="cyan"))
    
    m_choice = Prompt.ask("\nSelect model", choices=["1"], default="1")
    m_map = {"1": "xgboost"}
    m_type = m_map[m_choice]
    
    start_date = Prompt.ask("Enter start date (YYYY-MM-DD)", default="2024-01-01")
    end_date = Prompt.ask("Enter end date (YYYY-MM-DD)", default="2025-09-30")
    
    retrain_choice = Prompt.ask("Retrain before backtest?", choices=["y", "n"], default="n")
    retrain = True if retrain_choice.lower() == 'y' else False
    
    cmd = ["uv", "run", "python", "scripts/ml_backtest.py", "--model", m_type, "--start", start_date, "--end", end_date]
    if retrain:
        cmd.append("--retrain")
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

def train_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Model Training Lab[/bold cyan]", border_style="cyan"))
    
    m_choice = Prompt.ask("\nSelect model", choices=["1"], default="1")
    m_map = {"1": "xgboost"}
    m_type = m_map[m_choice]
    
    target = Prompt.ask("Enter target Win Rate (e.g. 0.85)", default="0.80")
    max_iter = IntPrompt.ask("Enter max optimization iterations", default=3)
    
    cmd = ["uv", "run", "python", "scripts/train_model.py", "--type", m_type, "--target", target, "--max-iter", str(max_iter)]
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted by user.[/dim]")
