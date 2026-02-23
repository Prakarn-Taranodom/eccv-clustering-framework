"""Data loading utilities for ECCV framework."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional

try:
    from aeon.datasets import load_classification
    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False
    print("Warning: aeon not available. UCR dataset loading will be limited.")


class DataLoader:
    """Load and manage datasets for ECCV experiments."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to data directory. If None, uses default.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / "datasets"
        else:
            self.data_dir = Path(data_dir)
    
    def load_ucr_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load UCR time series dataset.
        
        Args:
            name: Dataset name (e.g., 'GunPoint', 'ECG200')
            
        Returns:
            Tuple of (X, y) where X is data and y is labels
        """
        if not AEON_AVAILABLE:
            raise ImportError(
                "aeon is required for UCR dataset loading. "
                "Install with: pip install aeon"
            )
        X, y = load_classification(name)
        return X, y
    
    def load_csv_dataset(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame containing the dataset
        """
        return pd.read_csv(filepath)
    
    def get_timeseries_datasets(self) -> list:
        """Get list of available time-series datasets."""
        ts_dir = self.data_dir / "timeseries"
        if ts_dir.exists():
            return [f.stem for f in ts_dir.glob("*.csv")]
        return []
    
    def get_non_timeseries_datasets(self) -> list:
        """Get list of available non-time-series datasets."""
        nts_dir = self.data_dir / "non_timeseries"
        if nts_dir.exists():
            return [f.stem for f in nts_dir.glob("*.csv")]
        return []
