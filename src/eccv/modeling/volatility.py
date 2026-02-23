"""ARIMA-GARCH modeling for conditional volatility extraction."""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ConditionalVolatilityExtractor:
    """Extract conditional volatility using ARIMA-GARCH modeling."""
    
    def __init__(self, arima_order: Tuple[int, int, int] = (1, 0, 1),
                 garch_p: int = 1, garch_q: int = 1):
        """
        Initialize CV extractor.
        
        Args:
            arima_order: (p, d, q) for ARIMA model
            garch_p: GARCH p parameter
            garch_q: GARCH q parameter
        """
        self.arima_order = arima_order
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.arima_model = None
        self.garch_model = None
    
    def fit_arima(self, data: np.ndarray) -> np.ndarray:
        """
        Fit ARIMA model and return residuals.
        
        Args:
            data: Time series data
            
        Returns:
            ARIMA residuals
        """
        try:
            model = SARIMAX(data, order=self.arima_order)
            self.arima_model = model.fit(disp=False)
            residuals = self.arima_model.resid
            return residuals
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            return data - np.mean(data)
    
    def fit_garch(self, residuals: np.ndarray) -> np.ndarray:
        """
        Fit GARCH model and extract conditional volatility.
        
        Args:
            residuals: ARIMA residuals
            
        Returns:
            Conditional volatility series
        """
        try:
            # Scale residuals to avoid numerical issues
            residuals_scaled = residuals * 100
            
            model = arch_model(residuals_scaled, 
                             vol='Garch', 
                             p=self.garch_p, 
                             q=self.garch_q)
            self.garch_model = model.fit(disp='off', show_warning=False)
            
            # Extract conditional volatility
            cond_vol = self.garch_model.conditional_volatility / 100
            return cond_vol
        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            return np.abs(residuals)
    
    def extract_cv(self, data: np.ndarray) -> np.ndarray:
        """
        Extract conditional volatility from time series.
        
        Args:
            data: Time series data
            
        Returns:
            Conditional volatility series
        """
        # Step 1: Fit ARIMA and get residuals
        residuals = self.fit_arima(data)
        
        # Step 2: Fit GARCH on residuals
        cond_vol = self.fit_garch(residuals)
        
        return cond_vol
    
    def extract_cv_batch(self, data_list: list) -> list:
        """
        Extract CV for multiple time series.
        
        Args:
            data_list: List of time series arrays
            
        Returns:
            List of conditional volatility arrays
        """
        cv_list = []
        for i, data in enumerate(data_list):
            try:
                cv = self.extract_cv(data)
                cv_list.append(cv)
            except Exception as e:
                print(f"Failed to extract CV for series {i}: {e}")
                cv_list.append(np.zeros_like(data))
        
        return cv_list


def extract_cv_features(X: np.ndarray, 
                       arima_order: Tuple[int, int, int] = (1, 0, 1),
                       garch_p: int = 1, 
                       garch_q: int = 1) -> np.ndarray:
    """
    Convenience function to extract CV features from dataset.
    
    Args:
        X: Dataset of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
        arima_order: ARIMA (p, d, q) parameters
        garch_p: GARCH p parameter
        garch_q: GARCH q parameter
        
    Returns:
        CV features of same shape as input
    """
    extractor = ConditionalVolatilityExtractor(arima_order, garch_p, garch_q)
    
    # Handle different input shapes
    if X.ndim == 2:
        # (n_samples, n_timepoints)
        cv_features = np.array([extractor.extract_cv(x) for x in X])
    elif X.ndim == 3:
        # (n_samples, n_channels, n_timepoints)
        cv_features = np.array([[extractor.extract_cv(x[c]) 
                                for c in range(x.shape[0])] 
                               for x in X])
    else:
        raise ValueError(f"Unsupported input shape: {X.shape}")
    
    return cv_features
