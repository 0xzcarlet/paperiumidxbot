"""
Alternative Model: Gradient Descent + Supply/Demand Price Action
Uses logistic regression with SGD and price action patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class SupplyDemandZone:
    """Represents a supply or demand zone."""
    zone_type: str  # 'demand' or 'supply'
    price_low: float
    price_high: float
    strength: int  # Number of times tested
    created_date: pd.Timestamp
    last_test_date: pd.Timestamp
    broken: bool = False


class SupplyDemandDetector:
    """
    Detects supply and demand zones from price action.
    
    Supply zones: Areas where selling pressure overwhelmed buying
    Demand zones: Areas where buying pressure overwhelmed selling
    """
    
    def __init__(self, lookback: int = 20, threshold_pct: float = 0.02):
        self.lookback = lookback
        self.threshold_pct = threshold_pct
        self.cache = {} # ticker: {last_idx: int, zones: List[SupplyDemandZone]}
    
    def detect_zones(self, df: pd.DataFrame, ticker: str = "default") -> List[SupplyDemandZone]:
        """Detect supply and demand zones with simple caching."""
        if len(df) < self.lookback * 2:
            return []
            
        # Initialize or get cache
        if ticker not in self.cache:
            self.cache[ticker] = {'last_idx': self.lookback, 'zones': []}
        
        cache = self.cache[ticker]
        start_idx = cache['last_idx']
        zones = cache['zones']
        
        if start_idx >= len(df) - 5:
            return zones
        
        # Find swing highs (supply) and swing lows (demand)
        # Note: In a real incremental system, we'd only look at the new data.
        # For simplicity here, we iterate from start_idx.
        for i in range(start_idx, len(df) - 5):
            date = df.iloc[i]['date'] if 'date' in df.columns else pd.Timestamp.now()
            
            # Demand zone
            if self._is_demand_zone(df, i):
                zone = SupplyDemandZone(
                    zone_type='demand',
                    price_low=df.iloc[i-2:i+1]['low'].min(),
                    price_high=df.iloc[i-2:i+1]['high'].max(),
                    strength=1,
                    created_date=date,
                    last_test_date=date
                )
                zones.append(zone)
            
            # Supply zone
            if self._is_supply_zone(df, i):
                zone = SupplyDemandZone(
                    zone_type='supply',
                    price_low=df.iloc[i-2:i+1]['low'].min(),
                    price_high=df.iloc[i-2:i+1]['high'].max(),
                    strength=1,
                    created_date=date,
                    last_test_date=date
                )
                zones.append(zone)
        
        cache['last_idx'] = len(df) - 5
        cache['zones'] = self._merge_nearby_zones(zones)
        return cache['zones']
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """Find local maxima."""
        highs = df['high'].rolling(window, center=True).max()
        return df['high'] == highs
    
    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """Find local minima."""
        lows = df['low'].rolling(window, center=True).min()
        return df['low'] == lows
    
    def _is_demand_zone(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if index represents a demand zone formation."""
        if idx < 3 or idx >= len(df) - 2:
            return False
        
        # Look for: consolidation followed by strong up move
        base_range = df.iloc[idx-2:idx+1]
        breakout = df.iloc[idx+1:idx+3]
        
        # Consolidation: small range
        consolidation_range = (base_range['high'].max() - base_range['low'].min()) / base_range['close'].mean()
        
        # Strong breakout up
        breakout_move = (breakout['close'].iloc[-1] - base_range['close'].iloc[0]) / base_range['close'].iloc[0]
        
        return consolidation_range < 0.03 and breakout_move > 0.02
    
    def _is_supply_zone(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if index represents a supply zone formation."""
        if idx < 3 or idx >= len(df) - 2:
            return False
        
        # Look for: consolidation followed by strong down move
        base_range = df.iloc[idx-2:idx+1]
        breakdown = df.iloc[idx+1:idx+3]
        
        # Consolidation: small range
        consolidation_range = (base_range['high'].max() - base_range['low'].min()) / base_range['close'].mean()
        
        # Strong breakdown
        breakdown_move = (breakdown['close'].iloc[-1] - base_range['close'].iloc[0]) / base_range['close'].iloc[0]
        
        return consolidation_range < 0.03 and breakdown_move < -0.02
    
    def _merge_nearby_zones(self, zones: List[SupplyDemandZone], threshold: float = 0.02) -> List[SupplyDemandZone]:
        """Merge zones that are close to each other."""
        if not zones:
            return []
        
        merged = []
        zones = sorted(zones, key=lambda z: z.price_low)
        
        current = zones[0]
        for zone in zones[1:]:
            if zone.zone_type == current.zone_type:
                overlap = (current.price_high - zone.price_low) / zone.price_low
                if overlap > -threshold:
                    # Merge zones
                    current = SupplyDemandZone(
                        zone_type=current.zone_type,
                        price_low=min(current.price_low, zone.price_low),
                        price_high=max(current.price_high, zone.price_high),
                        strength=current.strength + zone.strength,
                        created_date=min(current.created_date, zone.created_date),
                        last_test_date=max(current.last_test_date, zone.last_test_date)
                    )
                else:
                    merged.append(current)
                    current = zone
            else:
                merged.append(current)
                current = zone
        
        merged.append(current)
        return merged
    
    def score_price_action(self, df: pd.DataFrame, current_price: float, ticker: str = "default") -> Dict:
        """
        Score current price action relative to supply/demand zones.
        """
        zones = self.detect_zones(df, ticker)
        
        demand_zones = [z for z in zones if z.zone_type == 'demand' and not z.broken]
        supply_zones = [z for z in zones if z.zone_type == 'supply' and not z.broken]
        
        score = 0.0
        nearest_demand = None
        nearest_supply = None
        
        # Find nearest demand zone below current price
        below_demands = [z for z in demand_zones if z.price_high < current_price]
        if below_demands:
            nearest_demand = max(below_demands, key=lambda z: z.price_high)
            distance_pct = (current_price - nearest_demand.price_high) / current_price
            
            # Near demand zone = bullish
            if distance_pct < 0.02:
                score += 0.5 + (0.5 * nearest_demand.strength / 3)
            elif distance_pct < 0.05:
                score += 0.3
        
        # Find nearest supply zone above current price
        above_supplies = [z for z in supply_zones if z.price_low > current_price]
        if above_supplies:
            nearest_supply = min(above_supplies, key=lambda z: z.price_low)
            distance_pct = (nearest_supply.price_low - current_price) / current_price
            
            # Near supply zone = bearish resistance
            if distance_pct < 0.02:
                score -= 0.3
            elif distance_pct < 0.05:
                score -= 0.1
        
        return {
            'score': np.clip(score, -1, 1),
            'nearest_demand': nearest_demand,
            'nearest_supply': nearest_supply,
            'demand_zones': len(demand_zones),
            'supply_zones': len(supply_zones)
        }


class GradientDescentModel:
    """
    Logistic regression with Stochastic Gradient Descent for price prediction.
    Learns which patterns lead to positive next-day returns.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100, batch_size: int = 32):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0.0
        self.feature_names = []
        self.fitted = False
    
    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create features for the model."""
        features = pd.DataFrame(index=df.index)
        
        # Price action features
        features['return_1d'] = df['close'].pct_change()
        features['return_3d'] = df['close'].pct_change(3)
        features['return_5d'] = df['close'].pct_change(5)
        
        # Candle patterns
        features['body_size'] = (df['close'] - df['open']) / df['open']
        features['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        features['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Range features
        features['daily_range'] = (df['high'] - df['low']) / df['close']
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Volume features
        features['volume_change'] = df['volume'].pct_change()
        features['rel_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Trend features
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        features['price_sma20_ratio'] = df['close'] / sma_20 - 1
        features['sma20_sma50_ratio'] = sma_20 / sma_50 - 1
        
        # Momentum
        features['rsi'] = self._calculate_rsi(df['close']) / 100 - 0.5  # Center at 0
        
        # Volatility
        features['volatility'] = df['close'].pct_change().rolling(20).std()
        
        self.feature_names = features.columns.tolist()
        
        # Robust handling of NaNs: forward fill then fill remaining with 0
        return features.ffill().fillna(0).values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(span=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, df: pd.DataFrame, warm_start: bool = False) -> Dict:
        """
        Train the model using gradient descent.
        
        Args:
            df: Price data DataFrame
            warm_start: Whether to keep existing weights as starting point
            
        Returns:
            Training metrics
        """
        X = self._create_features(df)
        
        # Target: next day positive return
        y = (df['close'].shift(-1) > df['close']).astype(int).values
        
        # Remove last row (no target) and NaN rows
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        valid_mask[-1] = False  # Last row has no target
        
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 50:
            return {'error': 'Insufficient data'}
        
        # Ensure X is clean before calculating stats to avoid NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        
        X = (X - self.feature_mean) / self.feature_std
        
        # Initialize weights if not warm starting or if they don't exist
        n_features = X.shape[1]
        if not warm_start or self.weights is None:
            self.weights = np.random.randn(n_features) * 0.01
            self.bias = 0.0
        else:
            logger.info("Retraining existing GD weights (Warm Start)...")
        
        # Gradient descent
        n_samples = len(X)
        losses = []
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward pass
                z = X_batch @ self.weights + self.bias
                y_pred = self._sigmoid(z)
                
                # Binary cross-entropy loss
                epsilon = 1e-7
                loss = -np.mean(y_batch * np.log(y_pred + epsilon) + 
                               (1 - y_batch) * np.log(1 - y_pred + epsilon))
                epoch_loss += loss
                
                # Backward pass (gradients)
                error = y_pred - y_batch
                dw = (X_batch.T @ error) / len(y_batch)
                db = np.mean(error)
                
                # Update weights
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            losses.append(epoch_loss / (n_samples // self.batch_size + 1))
        
        self.fitted = True
        
        # Calculate final accuracy
        y_pred = self._sigmoid(X @ self.weights + self.bias) > 0.5
        accuracy = np.mean(y_pred == y)
        
        return {
            'accuracy': accuracy,
            'final_loss': losses[-1],
            'n_samples': n_samples
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive return."""
        if not self.fitted:
            return np.zeros(len(df))
        
        X = self._create_features(df)
        
        # Ensure X is clean before subtraction to avoid RuntimeWarning
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X = (X - self.feature_mean) / (self.feature_std + 1e-8)
        
        probs = self._sigmoid(X @ self.weights + self.bias)
        return probs
    
    def predict_latest(self, df: pd.DataFrame) -> float:
        """Predict for the latest data point."""
        probs = self.predict(df)
        return probs[-1] if len(probs) > 0 else 0.5
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on weight magnitude."""
        if self.weights is None:
            return {}
        
        importance = np.abs(self.weights)
        importance = importance / importance.sum()
        
        return dict(zip(self.feature_names, importance))
    
    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'feature_mean': self.feature_mean,
                'feature_std': self.feature_std,
                'feature_names': self.feature_names,
                'fitted': self.fitted
            }, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
            self.feature_mean = data['feature_mean']
            self.feature_std = data['feature_std']
            self.feature_names = data['feature_names']
            self.fitted = data['fitted']


class SupplyDemandModel:
    """
    Combined model using Gradient Descent + Supply/Demand analysis.
    """
    
    def __init__(self):
        self.gd_model = GradientDescentModel(learning_rate=0.01, n_epochs=100)
        self.sd_detector = SupplyDemandDetector()
        self.gd_weight = 0.6
        self.sd_weight = 0.4
    
    def train(self, df: pd.DataFrame, warm_start: bool = False) -> Dict:
        """Train the gradient descent component."""
        return self.gd_model.fit(df, warm_start=warm_start)
    
    def predict(self, df: pd.DataFrame, ticker: str = "default") -> pd.DataFrame:
        """
        Generate predictions using both models.
        """
        results = df.copy()
        
        # Gradient descent predictions
        gd_probs = self.gd_model.predict(df)
        gd_scores = (gd_probs - 0.5) * 2  # Convert to -1 to 1
        
        # Supply/Demand scores (for each row)
        sd_scores = []
        for i in range(len(df)):
            if i < 50:
                sd_scores.append(0)
            else:
                historical = df.iloc[:i+1]
                current_price = df.iloc[i]['close']
                sd_result = self.sd_detector.score_price_action(historical, current_price, ticker)
                sd_scores.append(sd_result['score'])
        
        sd_scores = np.array(sd_scores)
        
        # Combined score
        combined_score = self.gd_weight * gd_scores + self.sd_weight * sd_scores
        
        results['gd_score'] = gd_scores
        results['sd_score'] = sd_scores
        results['combined_score'] = combined_score
        
        return results
    
    def predict_latest(self, df: pd.DataFrame, ticker: str = "default") -> Dict:
        """Get prediction for latest data point."""
        gd_prob = self.gd_model.predict_latest(df)
        gd_score = (gd_prob - 0.5) * 2
        
        current_price = df.iloc[-1]['close']
        sd_result = self.sd_detector.score_price_action(df, current_price, ticker)
        
        combined = self.gd_weight * gd_score + self.sd_weight * sd_result['score']
        
        return {
            'gd_score': gd_score,
            'gd_probability': gd_prob,
            'sd_score': sd_result['score'],
            'combined_score': combined,
            'signal': 'BUY' if combined > 0.3 else ('SELL' if combined < -0.3 else 'HOLD'),
            'nearest_demand': sd_result['nearest_demand'],
            'nearest_supply': sd_result['nearest_supply']
        }
    
    def save(self, path: str):
        """Save model."""
        self.gd_model.save(path)
    
    def load(self, path: str):
        """Load model."""
        self.gd_model.load(path)
