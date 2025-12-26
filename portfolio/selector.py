"""
Portfolio Selector Module
Top-10 stock selection with sector diversification
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from data.fetcher import get_sector_mapping

logger = logging.getLogger(__name__)


class PortfolioSelector:
    """
    Selects top stocks from signal rankings with diversification constraints.
    """
    
    def __init__(self, config=None):
        """
        Initialize portfolio selector.
        
        Args:
            config: PortfolioConfig object (optional)
        """
        if config:
            self.max_positions = config.max_positions
            self.max_sector_exposure = config.max_sector_exposure
            self.min_avg_volume = config.min_avg_volume
        else:
            self.max_positions = 10
            self.max_sector_exposure = 0.30
            self.min_avg_volume = 1_000_000
        
        self.sector_mapping = get_sector_mapping()
    
    def select_top_stocks(
        self,
        rankings: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        n_stocks: int = 10,
        signal_filter: str = 'BUY'
    ) -> pd.DataFrame:
        """
        Select top N stocks with diversification constraints.
        
        Args:
            rankings: DataFrame from SignalCombiner.rank_stocks()
            price_data: Dictionary with price history per ticker
            n_stocks: Number of stocks to select
            signal_filter: Signal type to filter
            
        Returns:
            DataFrame with selected stocks
        """
        if rankings.empty:
            return pd.DataFrame()
        
        df = rankings.copy()
        
        # Filter by signal
        if signal_filter:
            df = df[df['signal'] == signal_filter]
        
        # Filter by liquidity
        df = self._apply_liquidity_filter(df, price_data)
        
        # Sort by score
        df = df.sort_values('composite_score', ascending=False)
        
        # Apply sector diversification
        selected = self._apply_sector_diversification(df, n_stocks)
        
        # Add metadata
        selected = self._add_selection_metadata(selected)
        
        return selected
    
    def _apply_liquidity_filter(
        self,
        df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Filter out illiquid stocks."""
        liquid_tickers = []
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            
            if ticker not in price_data:
                continue
            
            ticker_data = price_data[ticker]
            
            # Check average volume
            avg_volume = ticker_data['volume'].tail(20).mean()
            
            if avg_volume >= self.min_avg_volume:
                liquid_tickers.append(ticker)
        
        return df[df['ticker'].isin(liquid_tickers)]
    
    def _apply_sector_diversification(
        self,
        df: pd.DataFrame,
        n_stocks: int
    ) -> pd.DataFrame:
        """
        Select stocks with sector diversification.
        Maximum stocks per sector based on max_sector_exposure.
        """
        max_per_sector = max(1, int(n_stocks * self.max_sector_exposure))
        
        selected = []
        sector_count = {}
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            sector = self.sector_mapping.get(ticker, 'Other')
            
            # Check sector limit
            if sector_count.get(sector, 0) >= max_per_sector:
                continue
            
            selected.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1
            
            if len(selected) >= n_stocks:
                break
        
        return pd.DataFrame(selected)
    
    def _add_selection_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sector and rank information."""
        df = df.copy()
        
        df['sector'] = df['ticker'].map(
            lambda x: self.sector_mapping.get(x, 'Other')
        )
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_sector_breakdown(self, selected: pd.DataFrame) -> Dict[str, int]:
        """Get sector distribution of selected stocks."""
        if 'sector' not in selected.columns:
            selected['sector'] = selected['ticker'].map(
                lambda x: self.sector_mapping.get(x, 'Other')
            )
        
        return selected['sector'].value_counts().to_dict()
    
    def filter_by_momentum_regime(
        self,
        df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter candidates based on market regime.
        In bearish regime, prefer defensive stocks.
        """
        # Calculate market trend (using IHSG/^JKSE)
        if market_data is not None and len(market_data) > 20:
            market_return_20d = market_data['close'].iloc[-1] / market_data['close'].iloc[-20] - 1
            
            if market_return_20d < -0.05:  # Bearish regime
                # Prefer defensive sectors
                defensive = ['Consumer', 'Healthcare', 'Telecom', 'Infrastructure']
                df = df.copy()
                df['is_defensive'] = df['ticker'].map(
                    lambda x: self.sector_mapping.get(x, '') in defensive
                )
                # Boost defensive scores
                df.loc[df['is_defensive'], 'composite_score'] *= 1.2
                df = df.sort_values('composite_score', ascending=False)
        
        return df


def select_top_10(
    rankings: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    config=None
) -> pd.DataFrame:
    """
    Convenience function to select top 10 stocks.
    
    Args:
        rankings: Signal rankings DataFrame
        price_data: Price history dictionary
        config: Optional configuration
        
    Returns:
        Top 10 selected stocks
    """
    selector = PortfolioSelector(config)
    return selector.select_top_stocks(rankings, price_data, n_stocks=10)
