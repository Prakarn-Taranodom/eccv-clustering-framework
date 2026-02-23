"""Clustering algorithms wrapper for ECCV framework."""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import PowerTransformer
from typing import Optional, Union

try:
    from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

try:
    from aeon.clustering import TimeSeriesCLARA, TimeSeriesCLARANS
    AEON_AVAILABLE = True
except (ImportError, AttributeError):
    AEON_AVAILABLE = False


class ClusteringPipeline:
    """Unified interface for various clustering algorithms."""
    
    ALGORITHMS = {
        'kmeans': KMeans,
        'spectral': SpectralClustering,
        'agglomerative': AgglomerativeClustering,
        'dbscan': DBSCAN,
    }
    
    if TSLEARN_AVAILABLE:
        ALGORITHMS.update({
            'ts_kmeans': TimeSeriesKMeans,
            'kshape': KShape,
            'kernel_kmeans': KernelKMeans,
        })
    
    if AEON_AVAILABLE:
        ALGORITHMS.update({
            'clara': TimeSeriesCLARA,
            'clarans': TimeSeriesCLARANS,
        })
    
    def __init__(self, algorithm: str = 'kmeans', 
                 n_clusters: int = 3,
                 standardize: bool = True,
                 **kwargs):
        """
        Initialize clustering pipeline.
        
        Args:
            algorithm: Name of clustering algorithm
            n_clusters: Number of clusters
            standardize: Whether to standardize data before clustering (Yeo-Johnson)
            **kwargs: Additional parameters for the clustering algorithm
        """
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.standardize = standardize
        self.scaler = PowerTransformer(method='yeo-johnson', standardize=True) if standardize else None
        self.model = None
        self.labels_ = None
        
        # Initialize clustering model
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                           f"Available: {list(self.ALGORITHMS.keys())}")
        
        # Set default parameters
        if self.algorithm in ['kmeans', 'ts_kmeans', 'kshape', 'kernel_kmeans', 
                             'clara', 'clarans', 'spectral', 'agglomerative']:
            kwargs.setdefault('n_clusters', n_clusters)
        
        self.model = self.ALGORITHMS[self.algorithm](**kwargs)
    
    def fit(self, X: np.ndarray) -> 'ClusteringPipeline':
        """
        Fit clustering model.
        
        Args:
            X: Data to cluster
            
        Returns:
            Self
        """
        # Reshape if needed for sklearn algorithms
        X_processed = self._preprocess(X)
        
        # Fit model
        self.model.fit(X_processed)
        
        # Get labels
        if hasattr(self.model, 'labels_'):
            self.labels_ = self.model.labels_
        elif hasattr(self.model, 'predict'):
            self.labels_ = self.model.predict(X_processed)
        else:
            raise AttributeError("Model doesn't have labels_ or predict method")
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return cluster labels.
        
        Args:
            X: Data to cluster
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data for clustering."""
        # Handle 3D time series data
        if X.ndim == 3:
            # For sklearn algorithms, flatten to 2D
            if self.algorithm in ['kmeans', 'spectral', 'agglomerative', 'dbscan']:
                n_samples = X.shape[0]
                X = X.reshape(n_samples, -1)
        
        # Standardize if requested
        if self.standardize and self.scaler is not None:
            if X.ndim == 2:
                X = self.scaler.fit_transform(X)
            elif X.ndim == 3:
                # Standardize each time series
                n_samples, n_channels, n_timepoints = X.shape
                X_reshaped = X.reshape(-1, n_timepoints)
                X_transformed = self.scaler.fit_transform(X_reshaped.T).T
                X = X_transformed.reshape(n_samples, n_channels, n_timepoints)
        
        return X


def cluster_with_cv(X: np.ndarray, 
                   cv_features: np.ndarray,
                   algorithm: str = 'kmeans',
                   n_clusters: int = 3,
                   **kwargs) -> np.ndarray:
    """
    Cluster data using conditional volatility features.
    
    Args:
        X: Original data (not used, kept for API consistency)
        cv_features: Conditional volatility features
        algorithm: Clustering algorithm name
        n_clusters: Number of clusters
        **kwargs: Additional clustering parameters
        
    Returns:
        Cluster labels
    """
    pipeline = ClusteringPipeline(algorithm, n_clusters, **kwargs)
    return pipeline.fit_predict(cv_features)


def cluster_without_cv(X: np.ndarray,
                      algorithm: str = 'kmeans',
                      n_clusters: int = 3,
                      **kwargs) -> np.ndarray:
    """
    Cluster data using original features (baseline).
    
    Args:
        X: Original data
        algorithm: Clustering algorithm name
        n_clusters: Number of clusters
        **kwargs: Additional clustering parameters
        
    Returns:
        Cluster labels
    """
    pipeline = ClusteringPipeline(algorithm, n_clusters, **kwargs)
    return pipeline.fit_predict(X)
