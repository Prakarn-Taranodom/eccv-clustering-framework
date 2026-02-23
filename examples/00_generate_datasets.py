"""
Generate Synthetic Datasets for ECCV Benchmark
Creates 40 datasets (20 time-series + 20 non-time-series) for testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification, make_blobs
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def generate_timeseries_dataset(n_samples=100, n_timepoints=100, 
                                n_clusters=3, noise_level=0.1, 
                                pattern_type='sine'):
    """
    Generate synthetic time series dataset.
    
    Args:
        n_samples: Number of time series
        n_timepoints: Length of each series
        n_clusters: Number of clusters
        noise_level: Amount of noise
        pattern_type: 'sine', 'trend', 'seasonal', 'random_walk'
        
    Returns:
        X: Time series data (n_samples, n_timepoints)
        y: Cluster labels
    """
    X = []
    y = []
    
    samples_per_cluster = n_samples // n_clusters
    t = np.linspace(0, 4*np.pi, n_timepoints)
    
    for cluster_id in range(n_clusters):
        for _ in range(samples_per_cluster):
            if pattern_type == 'sine':
                # Sine wave with different frequencies
                freq = 1 + cluster_id * 0.5
                series = np.sin(freq * t) + np.random.normal(0, noise_level, n_timepoints)
                
            elif pattern_type == 'trend':
                # Linear trend with different slopes
                slope = 0.01 * (cluster_id + 1)
                series = slope * np.arange(n_timepoints) + np.random.normal(0, noise_level, n_timepoints)
                
            elif pattern_type == 'seasonal':
                # Seasonal pattern
                period = 20 + cluster_id * 10
                series = np.sin(2 * np.pi * np.arange(n_timepoints) / period)
                series += np.random.normal(0, noise_level, n_timepoints)
                
            elif pattern_type == 'random_walk':
                # Random walk with different drift
                drift = 0.01 * cluster_id
                series = np.cumsum(np.random.normal(drift, 0.1, n_timepoints))
                series += np.random.normal(0, noise_level, n_timepoints)
            
            X.append(series)
            y.append(cluster_id)
    
    return np.array(X), np.array(y)


def generate_non_timeseries_dataset(n_samples=100, n_features=20, 
                                   n_clusters=3, noise_level=0.1):
    """
    Generate synthetic non-time-series dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_clusters: Number of clusters
        noise_level: Amount of noise
        
    Returns:
        X: Feature data (n_samples, n_features)
        y: Cluster labels
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=noise_level * 10,
        random_state=42
    )
    
    return X, y


def generate_all_datasets(output_dir='datasets', n_datasets_per_type=20):
    """
    Generate all benchmark datasets.
    
    Args:
        output_dir: Output directory
        n_datasets_per_type: Number of datasets per type (TS and non-TS)
    """
    output_path = Path(output_dir)
    ts_path = output_path / 'timeseries'
    nts_path = output_path / 'non_timeseries'
    
    ts_path.mkdir(parents=True, exist_ok=True)
    nts_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating Synthetic Datasets for ECCV Benchmark")
    print("="*60)
    
    # Generate time-series datasets
    print(f"\nGenerating {n_datasets_per_type} time-series datasets...")
    
    patterns = ['sine', 'trend', 'seasonal', 'random_walk']
    
    for i in range(n_datasets_per_type):
        # Vary parameters
        n_samples = np.random.randint(50, 200)
        n_timepoints = np.random.randint(50, 150)
        n_clusters = np.random.randint(2, 6)
        noise_level = np.random.uniform(0.05, 0.3)
        pattern = patterns[i % len(patterns)]
        
        X, y = generate_timeseries_dataset(
            n_samples=n_samples,
            n_timepoints=n_timepoints,
            n_clusters=n_clusters,
            noise_level=noise_level,
            pattern_type=pattern
        )
        
        # Save as CSV
        df = pd.DataFrame(X)
        df['label'] = y
        
        filename = ts_path / f'dataset_ts_{chr(65+i)}.csv'  # A, B, C, ...
        df.to_csv(filename, index=False)
        
        print(f"  [OK] {filename.name}: {n_samples} samples, {n_timepoints} timepoints, {n_clusters} clusters ({pattern})")
    
    # Generate non-time-series datasets
    print(f"\nGenerating {n_datasets_per_type} non-time-series datasets...")
    
    for i in range(n_datasets_per_type):
        # Vary parameters
        n_samples = np.random.randint(50, 200)
        n_features = np.random.randint(10, 50)
        n_clusters = np.random.randint(2, 6)
        noise_level = np.random.uniform(0.05, 0.3)
        
        X, y = generate_non_timeseries_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_clusters=n_clusters,
            noise_level=noise_level
        )
        
        # Save as CSV
        df = pd.DataFrame(X)
        df['label'] = y
        
        filename = nts_path / f'dataset_nts_{chr(65+i)}.csv'  # A, B, C, ...
        df.to_csv(filename, index=False)
        
        print(f"  [OK] {filename.name}: {n_samples} samples, {n_features} features, {n_clusters} clusters")
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Generated {n_datasets_per_type * 2} datasets")
    print(f"  - Time-series: {ts_path}")
    print(f"  - Non-time-series: {nts_path}")
    print("="*60)


def main():
    """Generate datasets."""
    generate_all_datasets(
        output_dir='datasets',
        n_datasets_per_type=20
    )


if __name__ == "__main__":
    main()
