"""
Machine Learning Model Module
XGBoost-based prediction with daily self-refinement
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import pickle
import os
from datetime import datetime
import logging

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from .features import FeatureEngineer
from .supply_demand_model import SupplyDemandModel

logger = logging.getLogger(__name__)


class TradingModel:
    """
    XGBoost-based trading model with rolling window training.
    Implements daily self-refinement (EOD model update).
    """
    
    def __init__(self, config=None):
        """
        Initialize trading model.
        
        Args:
            config: MLConfig object (optional)
        """
        if config:
            self.training_window = config.training_window
            self.min_training_samples = getattr(config, 'min_training_samples', 60)
            self.n_estimators = config.n_estimators
            self.max_depth = config.max_depth
            self.learning_rate = config.learning_rate
            self.min_child_weight = config.min_child_weight
        else:
            self.training_window = 252
            self.min_training_samples = 40
            self.n_estimators = 100
            self.max_depth = 5
            self.learning_rate = 0.1
            self.min_child_weight = 3
        
        self.feature_engineer = FeatureEngineer(config)
        self.model = None
        self.feature_names = None
        self.last_trained = None
        self.performance_history = []
    
    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with configured parameters."""
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    
    def train(
        self, 
        df: pd.DataFrame,
        validate: bool = True
    ) -> Dict[str, float]:
        """
        Train the model on historical data.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            validate: Whether to perform validation
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training ML model...")
        
        # Create features
        X, y = self.feature_engineer.create_features(df)
        
        # Clean data (handle inf and nan)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)
        
        if len(X) < self.min_training_samples:
            logger.warning(f"Insufficient data: {len(X)} < {self.min_training_samples}")
            return {}
        
        # Use rolling window (adaptive)
        current_window = min(len(X), self.training_window)
        X_train = X.iloc[-current_window:]
        y_train = y.iloc[-current_window:]
        
        self.feature_names = X_train.columns.tolist()
        
        # Train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.last_trained = datetime.now()
        
        metrics = {
            'train_samples': len(X_train),
            'features': len(self.feature_names)
        }
        
        # Validate if requested
        if validate:
            val_metrics = self._validate(X_train, y_train)
            metrics.update(val_metrics)
        
        logger.info(f"Model trained: {metrics}")
        
        return metrics
    
    def _validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            
        Returns:
            Dictionary of validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        accuracies = []
        precisions = []
        
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self._create_model()
            model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_val)
            
            accuracies.append(accuracy_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
        
        return {
            'cv_accuracy': np.mean(accuracies),
            'cv_accuracy_std': np.std(accuracies),
            'cv_precision': np.mean(precisions)
        }
    
    def daily_update(
        self, 
        df: pd.DataFrame,
        new_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Daily model update (self-refinement).
        
        Uses rolling window: drops oldest day, adds newest day.
        This is the key to the model's ability to adapt to changing market conditions.
        
        Args:
            df: Full historical DataFrame
            new_data: Optional new data to append
            
        Returns:
            Updated training metrics
        """
        logger.info("Performing daily model update...")
        
        if new_data is not None:
            df = pd.concat([df, new_data], ignore_index=True)
        
        # Retrain with latest data
        metrics = self.train(df, validate=True)
        
        if metrics:
            self.performance_history.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                **metrics
            })
        
        return metrics
    
    def predict(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            
        Returns:
            Tuple of (class predictions, probability predictions)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = self.feature_engineer.prepare_inference_features(df)
        
        # Align features with training features
        missing_features = set(self.feature_names) - set(X.columns)
        for f in missing_features:
            X[f] = 0
        
        X = X[self.feature_names]
        
        # Predict
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (up)
        
        return y_pred, y_proba
    
    def predict_latest(
        self, 
        df: pd.DataFrame
    ) -> Tuple[int, float]:
        """
        Predict for the latest (most recent) data point.
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            Tuple of (direction prediction, probability)
        """
        y_pred, y_proba = self.predict(df)
        return y_pred[-1], y_proba[-1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        
        return self.feature_engineer.get_feature_importance(
            self.model, 
            self.feature_names
        )
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: File path for model
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'last_trained': self.last_trained,
            'performance_history': self.performance_history,
            'config': {
                'training_window': self.training_window,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: File path for model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.last_trained = model_data['last_trained']
        self.performance_history = model_data.get('performance_history', [])
        
        logger.info(f"Model loaded from {path}, last trained: {self.last_trained}")


class EnsembleModel:
    """
    High-performance ensemble of XGBoost and SupplyDemandModel.
    Matches the architecture validated in backtesting.
    """
    
    def __init__(self, config=None, model_dir: str = 'models'):
        self.config = config
        # If passed global config, use the ml sub-config for TradingModel
        ml_config = getattr(config, 'ml', config)
        self.xgb_model = TradingModel(ml_config)
        self.sd_model = SupplyDemandModel()
        self.model_dir = model_dir
        
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train ensemble models using pooled data.
        Note: The backtest showed global pooling is superior.
        """
        # XGBoost training
        xgb_metrics = self.xgb_model.train(df)
        
        # SD (Gradient Descent) training
        sd_metrics = self.sd_model.train(df)
        
        return {
            'xgb_trained': self.xgb_model.model is not None,
            'sd_trained': self.sd_model.gd_model.weights is not None,
            'xgb_metrics': xgb_metrics,
            'sd_metrics': sd_metrics
        }
    
    def predict_latest(self, df: pd.DataFrame, ticker: str = "default") -> Dict:
        """Ensemble prediction for latest data point."""
        scores = []
        
        # XGBoost contribution
        if self.xgb_model.model is not None:
            _, xgb_proba = self.xgb_model.predict_latest(df)
            scores.append((xgb_proba - 0.5) * 2)
            
        # SD contribution
        if self.sd_model.gd_model.weights is not None:
            sd_res = self.sd_model.predict_latest(df, ticker)
            scores.append(sd_res['combined_score'])
            
        if not scores:
            return {'combined_score': 0.0, 'signals': {}}
            
        # Weighted mean: if only one exists, it is 100% weight.
        # If both exist, we give them equal weight for now (Ensemble).
        combined_score = np.mean(scores)
        
        return {
            'combined_score': float(combined_score),
            'components': {
                'xgb': float(scores[0]) if len(scores) > 0 and self.xgb_model.model is not None else 0,
                'sd': float(scores[1]) if len(scores) > 1 else (float(scores[0]) if self.xgb_model.model is None else 0)
            }
        }

    def save(self, name: str = 'global_ensemble'):
        """Save both models."""
        xgb_path = os.path.join(self.model_dir, f"{name}_xgb.pkl")
        sd_path = os.path.join(self.model_dir, f"{name}_sd.pkl")
        
        self.xgb_model.save(xgb_path)
        self.sd_model.save(sd_path)
        logger.info(f"Ensemble saved to {self.model_dir}")
        
    def load(self, name: str = 'global_ensemble'):
        """Load both models, preferring champion versions if they exist."""
        # Try champion first, then fall back to provided name
        xgb_champ = os.path.join(self.model_dir, "global_xgb_champion.pkl")
        sd_champ = os.path.join(self.model_dir, "global_sd_champion.pkl")
        
        xgb_path = xgb_champ if os.path.exists(xgb_champ) else os.path.join(self.model_dir, f"{name}_xgb.pkl")
        sd_path = sd_champ if os.path.exists(sd_champ) else os.path.join(self.model_dir, f"{name}_sd.pkl")
        
        if os.path.exists(xgb_path):
            self.xgb_model.load(xgb_path)
            logger.info(f"XGBoost loaded from {xgb_path}")
        if os.path.exists(sd_path):
            self.sd_model.load(sd_path)
            logger.info(f"SD Model loaded from {sd_path}")
