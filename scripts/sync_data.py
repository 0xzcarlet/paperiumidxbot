#!/usr/bin/env python3
import sys
import os
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from data.fetcher import DataFetcher
from data.storage import DataStorage

console = Console()
logging.basicConfig(level=logging.INFO)

def sync_data():
    console.print("[bold cyan]Starting Universe Data Sync[/bold cyan]")
    storage = DataStorage(config.data.db_path)
    fetcher = DataFetcher(config.data.stock_universe)
    
    total_tickers = len(config.data.stock_universe)
    console.print(f"Syncing [bold]{total_tickers}[/bold] tickers (2 years history)...")
    
    # Use fetch_batch but track progress
    # fetch_batch uses yfinance's internal progress bar, but we'll wrap it to be sure
    data = fetcher.fetch_batch(days=730)
    
    if not data.empty:
        console.print(f"Upserting {len(data)} records to database...")
        count = storage.upsert_prices(data)
        console.print(f"[bold green]✓ Database Sync Complete. {count} records updated.[/bold green]")
    else:
        console.print("[bold red]✕ Data fetch failed or returned empty.[/bold red]")

if __name__ == "__main__":
    sync_data()
