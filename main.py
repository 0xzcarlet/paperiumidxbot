#!/usr/bin/env python3
"""
IHSG Quantitative Trading Model
Main Entry Point

Inspired by Jim Simons' Renaissance Technologies approach.
Designed for Indonesian stock market daily trading.

Usage:
    python main.py                    # Run full pipeline (fetch + signals + report)
    python main.py --mode fetch       # Fetch latest data only
    python main.py --mode signals     # Generate signals from existing data
    python main.py --mode backtest    # Run backtesting
    python main.py --mode schedule    # Start daily scheduler
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

from config import config
from data.fetcher import DataFetcher
from data.storage import DataStorage
from signals.combiner import SignalCombiner
from ml.model import TradingModel
from portfolio.selector import PortfolioSelector
from strategy.exit_manager import ExitManager
from strategy.position_sizer import PositionSizer
from reports.generator import ReportGenerator
from backtest.engine import BacktestEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading.log')
    ]
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self):
        self.config = config
        
        # Initialize components
        self.fetcher = DataFetcher(config.data.stock_universe)
        self.storage = DataStorage(config.data.db_path)
        self.signal_combiner = SignalCombiner(config)
        self.ml_model = TradingModel(config.ml)
        self.selector = PortfolioSelector(config.portfolio)
        self.exit_manager = ExitManager(config.exit)
        self.position_sizer = PositionSizer(config.portfolio)
        self.report_generator = ReportGenerator(config)
    
    def run_full_pipeline(self):
        """Run the complete daily pipeline."""
        logger.info("=" * 60)
        logger.info("IHSG Quantitative Trading System")
        logger.info("=" * 60)
        
        # Step 1: Fetch data
        logger.info("\n[1/5] Fetching market data...")
        self.fetch_data()
        
        # Step 2: Load and prepare data
        logger.info("\n[2/5] Loading data...")
        all_data = self.storage.get_prices()
        
        if all_data.empty:
            logger.error("No data available. Run with --mode fetch first.")
            return None
        
        logger.info(f"Loaded {len(all_data)} price records for {all_data['ticker'].nunique()} stocks")
        
        # Step 3: Generate signals
        logger.info("\n[3/5] Generating trading signals...")
        signals = self.generate_signals(all_data)
        
        if signals.empty:
            logger.warning("No signals generated.")
            return None
        
        # Step 4: Select top 10
        logger.info("\n[4/5] Selecting top 10 stocks...")
        top_10 = self.select_top_stocks(signals, all_data)
        
        # Step 5: Generate report
        logger.info("\n[5/5] Generating report...")
        if not top_10.empty:
            self.report_generator.generate_daily_report(
                top_10, signals, datetime.now()
            )
        
        return top_10
    
    def fetch_data(self, days: int = None):
        """Fetch and store market data."""
        days = days or config.data.lookback_days
        
        logger.info(f"Fetching {days} days of historical data...")
        data = self.fetcher.fetch_batch(days=days)
        
        if not data.empty:
            count = self.storage.upsert_prices(data)
            logger.info(f"Stored {count} price records")
        else:
            logger.warning("No data fetched")
        
        return data
    
    def generate_signals(self, all_data):
        """Generate signals for all stocks."""
        data_by_ticker = {}
        
        for ticker in all_data['ticker'].unique():
            ticker_data = all_data[all_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            if len(ticker_data) >= config.data.min_data_points:
                try:
                    signals = self.signal_combiner.calculate_signals(ticker_data)
                    data_by_ticker[ticker] = signals
                except Exception as e:
                    logger.warning(f"Signal calculation failed for {ticker}: {e}")
        
        rankings = self.signal_combiner.rank_stocks(data_by_ticker)
        return rankings
    
    def select_top_stocks(self, signals, all_data):
        """Select top 10 stocks from rankings."""
        # Prepare price data dictionary
        price_data = {}
        for ticker in signals['ticker'].unique():
            price_data[ticker] = all_data[all_data['ticker'] == ticker]
        
        # Select top 10
        top_10 = self.selector.select_top_stocks(signals, price_data, n_stocks=10)
        
        # Add exit levels
        if not top_10.empty:
            top_10 = self._add_exit_levels(top_10)
        
        return top_10
    
    def _add_exit_levels(self, stocks):
        """Add stop-loss and take-profit levels."""
        stocks = stocks.copy()
        
        for idx, row in stocks.iterrows():
            entry_price = row.get('close', row.get('entry_price', 0))
            atr = row.get('atr', entry_price * 0.02)
            
            levels = self.exit_manager.calculate_levels(entry_price, atr)
            
            stocks.loc[idx, 'stop_loss'] = levels.stop_loss
            stocks.loc[idx, 'take_profit'] = levels.take_profit
        
        return stocks
    
    def run_backtest(
        self,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100_000_000
    ):
        """Run backtesting."""
        # Parse dates
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        logger.info(f"Running backtest: {start_date.date()} to {end_date.date()}")
        
        # Load data
        all_data = self.storage.get_prices()
        
        if all_data.empty:
            logger.error("No data available. Run fetch first.")
            return None
        
        # Organize by ticker
        price_data = {}
        for ticker in all_data['ticker'].unique():
            price_data[ticker] = all_data[all_data['ticker'] == ticker].copy()
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest(
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        if results:
            engine.print_results(results)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='IHSG Quantitative Trading Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Full pipeline
  python main.py --mode fetch --days 365           # Fetch 1 year of data
  python main.py --mode signals                    # Generate signals only
  python main.py --mode backtest --start 2024-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'fetch', 'signals', 'backtest', 'schedule'],
        default='full',
        help='Operation mode'
    )
    parser.add_argument('--days', type=int, default=config.data.lookback_days, help='Days of data to fetch')
    parser.add_argument('--start', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100_000_000, help='Initial capital for backtest')
    
    args = parser.parse_args()
    
    system = TradingSystem()
    
    if args.mode == 'fetch':
        system.fetch_data(days=args.days)
    
    elif args.mode == 'signals':
        all_data = system.storage.get_prices()
        signals = system.generate_signals(all_data)
        
        if not signals.empty:
            print("\nTop 20 Signals:")
            print(signals[['ticker', 'composite_score', 'signal']].head(20).to_string())
    
    elif args.mode == 'backtest':
        system.run_backtest(
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital
        )
    
    elif args.mode == 'schedule':
        from scheduler.daily_job import DailyJobScheduler
        scheduler = DailyJobScheduler()
        print("Starting scheduler (Ctrl+C to stop)...")
        scheduler.schedule_daily("16:30")
        scheduler.start()
    
    else:  # full
        system.run_full_pipeline()


if __name__ == "__main__":
    main()
