"""Preprocessing utilities for time series data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
from typing import Union, Optional


class TimeSeriesPreprocessor:
    """Preprocess time series data before ARIMA-GARCH modeling."""
    
    def __init__(self, 
                 normalization: str = 'zscore',
                 handle_missing: str = 'interpolate',
                 outlier_threshold: float = 3.0):
        """
        Initialize preprocessor.
        
        Args:
            normalization: 'zscore', 'minmax', 'robust', or None
            handle_missing: 'interpolate', 'forward_fill', 'drop'
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.normalization = normalization
        self.handle_missing = handle_missing
        self.outlier_threshold = outlier_threshold
        self.scaler = None
        
        if normalization == 'zscore':
            self.scaler = StandardScaler()
        elif normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalization == 'robust':
            self.scaler = RobustScaler()
    
    def handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in time series."""
        if not np.any(np.isnan(data)):
            return data
        
        if self.handle_missing == 'interpolate':
            # Linear interpolation
            df = pd.DataFrame(data)
            return df.interpolate(method='linear', limit_direction='both').values
        elif self.handle_missing == 'forward_fill':
            df = pd.DataFrame(data)
            return df.fillna(method='ffill').fillna(method='bfill').values
        elif self.handle_missing == 'drop':
            return data[~np.isnan(data)]
        else:
            return data
    
    def detect_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using z-score method.
        
        Returns:
            Boolean array indicating outliers
        """
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return z_scores > self.outlier_threshold
    
    def remove_outliers(self, data: np.ndarray, replace_with: str = 'median') -> np.ndarray:
        """
        Remove or replace outliers.
        
        Args:
            data: Time series data
            replace_with: 'median', 'mean', or 'interpolate'
        """
        outliers = self.detect_outliers(data)
        
        if not np.any(outliers):
            return data
        
        data_clean = data.copy()
        
        if replace_with == 'median':
            data_clean[outliers] = np.median(data[~outliers])
        elif replace_with == 'mean':
            data_clean[outliers] = np.mean(data[~outliers])
        elif replace_with == 'interpolate':
            data_clean[outliers] = np.nan
            df = pd.DataFrame(data_clean)
            data_clean = df.interpolate(method='linear').values.flatten()
        
        return data_clean
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize time series data."""
        if self.scaler is None:
            return data
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            normalized = self.scaler.fit_transform(data).flatten()
        else:
            normalized = self.scaler.fit_transform(data)
        
        return normalized
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            data: Time series data
            
        Returns:
            Preprocessed data
        """
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Remove outliers
        data = self.remove_outliers(data)
        
        # Normalize
        data = self.normalize(data)
        
        return data
    
    def preprocess_batch(self, data_list: list) -> list:
        """Preprocess multiple time series."""
        return [self.preprocess(data) for data in data_list]


def preprocess_for_cv_extraction(X: np.ndarray,
                                 normalization: str = 'zscore',
                                 handle_missing: str = 'interpolate',
                                 outlier_threshold: float = 3.0) -> np.ndarray:
    """
    Convenience function for preprocessing before CV extraction.
    
    Args:
        X: Time series dataset
        normalization: Normalization method
        handle_missing: How to handle missing values
        outlier_threshold: Outlier detection threshold
        
    Returns:
        Preprocessed dataset
    """
    preprocessor = TimeSeriesPreprocessor(
        normalization=normalization,
        handle_missing=handle_missing,
        outlier_threshold=outlier_threshold
    )
    
    if X.ndim == 2:
        # (n_samples, n_timepoints)
        return np.array([preprocessor.preprocess(x) for x in X])
    elif X.ndim == 3:
        # (n_samples, n_channels, n_timepoints)
        return np.array([[preprocessor.preprocess(x[c]) 
                         for c in range(x.shape[0])] 
                        for x in X])
    else:
        raise ValueError(f"Unsupported shape: {X.shape}")
