"""
Download real classification datasets for ECCV testing
- 5 Time-Series datasets from UCR Archive
- 5 Non-Time-Series datasets from sklearn
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# UCR Time Series datasets (small, well-known)
UCR_DATASETS = [
    'GunPoint',        # 2 classes, 150 length, simple
    'Coffee',          # 2 classes, 286 length
    'ECG200',          # 2 classes, 96 length
    'ItalyPowerDemand', # 2 classes, 24 length
    'TwoLeadECG'       # 2 classes, 82 length
]

def download_ucr_datasets():
    """Download UCR time series datasets."""
    print("="*70)
    print("Downloading UCR Time-Series Datasets")
    print("="*70)
    
    try:
        from aeon.datasets import load_classification
    except ImportError:
        print("aeon not installed. Skipping UCR datasets.")
        print("Install with: pip install aeon")
        return []
    
    datasets_dir = project_root / 'datasets' / 'timeseries'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for name in UCR_DATASETS:
        try:
            print(f"Downloading {name}...", end=' ')
            X, y = load_classification(name)
            
            # Convert to 2D if needed
            if X.ndim == 3:
                n_samples, n_channels, n_timepoints = X.shape
                X_2d = X.reshape(n_samples, -1)
            else:
                X_2d = X
            
            # Create DataFrame
            df = pd.DataFrame(X_2d)
            df['label'] = y
            
            # Save
            filepath = datasets_dir / f'{name}.csv'
            df.to_csv(filepath, index=False)
            
            print(f"OK ({X.shape[0]} samples, {X.shape[-1]} length, {len(np.unique(y))} classes)")
            downloaded.append({
                'name': name,
                'samples': X.shape[0],
                'features': X.shape[-1],
                'classes': len(np.unique(y))
            })
            
        except Exception as e:
            print(f"FAILED: {e}")
    
    return downloaded


def download_sklearn_datasets():
    """Download sklearn classification datasets."""
    print("\n" + "="*70)
    print("Downloading Non-Time-Series Datasets (sklearn)")
    print("="*70)
    
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
    from sklearn.datasets import make_moons
    
    datasets_dir = project_root / 'datasets' / 'non_timeseries'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    # 1. Iris
    try:
        print("Loading Iris...", end=' ')
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.to_csv(datasets_dir / 'Iris.csv', index=False)
        print(f"OK ({iris.data.shape[0]} samples, {iris.data.shape[1]} features, 3 classes)")
        downloaded.append({'name': 'Iris', 'samples': iris.data.shape[0], 
                          'features': iris.data.shape[1], 'classes': 3})
    except Exception as e:
        print(f"FAILED: {e}")
    
    # 2. Wine
    try:
        print("Loading Wine...", end=' ')
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['label'] = wine.target
        df.to_csv(datasets_dir / 'Wine.csv', index=False)
        print(f"OK ({wine.data.shape[0]} samples, {wine.data.shape[1]} features, 3 classes)")
        downloaded.append({'name': 'Wine', 'samples': wine.data.shape[0], 
                          'features': wine.data.shape[1], 'classes': 3})
    except Exception as e:
        print(f"FAILED: {e}")
    
    # 3. Breast Cancer
    try:
        print("Loading Breast Cancer...", end=' ')
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['label'] = cancer.target
        df.to_csv(datasets_dir / 'BreastCancer.csv', index=False)
        print(f"OK ({cancer.data.shape[0]} samples, {cancer.data.shape[1]} features, 2 classes)")
        downloaded.append({'name': 'BreastCancer', 'samples': cancer.data.shape[0], 
                          'features': cancer.data.shape[1], 'classes': 2})
    except Exception as e:
        print(f"FAILED: {e}")
    
    # 4. Digits (subset)
    try:
        print("Loading Digits (subset)...", end=' ')
        digits = load_digits(n_class=5)  # Only 5 classes for speed
        df = pd.DataFrame(digits.data)
        df['label'] = digits.target
        df.to_csv(datasets_dir / 'Digits.csv', index=False)
        print(f"OK ({digits.data.shape[0]} samples, {digits.data.shape[1]} features, 5 classes)")
        downloaded.append({'name': 'Digits', 'samples': digits.data.shape[0], 
                          'features': digits.data.shape[1], 'classes': 5})
    except Exception as e:
        print(f"FAILED: {e}")
    
    # 5. Moons (synthetic but classic)
    try:
        print("Loading Moons...", end=' ')
        X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['label'] = y
        df.to_csv(datasets_dir / 'Moons.csv', index=False)
        print(f"OK ({X.shape[0]} samples, {X.shape[1]} features, 2 classes)")
        downloaded.append({'name': 'Moons', 'samples': X.shape[0], 
                          'features': X.shape[1], 'classes': 2})
    except Exception as e:
        print(f"FAILED: {e}")
    
    return downloaded


def clean_old_datasets():
    """Remove old synthetic datasets."""
    print("\n" + "="*70)
    print("Cleaning Old Synthetic Datasets")
    print("="*70)
    
    datasets_dir = project_root / 'datasets'
    
    # Remove old TS datasets
    ts_dir = datasets_dir / 'timeseries'
    if ts_dir.exists():
        old_files = list(ts_dir.glob('dataset_ts_*.csv'))
        for f in old_files:
            f.unlink()
            print(f"Removed: {f.name}")
    
    # Remove old NTS datasets
    nts_dir = datasets_dir / 'non_timeseries'
    if nts_dir.exists():
        old_files = list(nts_dir.glob('dataset_nts_*.csv'))
        for f in old_files:
            f.unlink()
            print(f"Removed: {f.name}")
    
    print("Old datasets cleaned")


def create_summary(ts_datasets, nts_datasets):
    """Create summary of downloaded datasets."""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    print("\nTime-Series Datasets:")
    if ts_datasets:
        df = pd.DataFrame(ts_datasets)
        print(df.to_string(index=False))
    else:
        print("  None downloaded")
    
    print("\nNon-Time-Series Datasets:")
    if nts_datasets:
        df = pd.DataFrame(nts_datasets)
        print(df.to_string(index=False))
    else:
        print("  None downloaded")
    
    print(f"\nTotal: {len(ts_datasets)} TS + {len(nts_datasets)} Non-TS = {len(ts_datasets) + len(nts_datasets)} datasets")
    
    # Save summary
    summary_file = project_root / 'datasets' / 'DATASETS_INFO.txt'
    with open(summary_file, 'w') as f:
        f.write("ECCV Framework - Real Datasets Information\n")
        f.write("="*70 + "\n\n")
        f.write("Time-Series Datasets (UCR Archive):\n")
        for ds in ts_datasets:
            f.write(f"  - {ds['name']}: {ds['samples']} samples, {ds['features']} features, {ds['classes']} classes\n")
        f.write("\nNon-Time-Series Datasets (sklearn):\n")
        for ds in nts_datasets:
            f.write(f"  - {ds['name']}: {ds['samples']} samples, {ds['features']} features, {ds['classes']} classes\n")
    
    print(f"\nSummary saved to: {summary_file}")


def main():
    print("\n" + "="*70)
    print("ECCV Framework - Real Dataset Downloader")
    print("="*70)
    
    # Clean old datasets
    clean_old_datasets()
    
    # Download new datasets
    ts_datasets = download_ucr_datasets()
    nts_datasets = download_sklearn_datasets()
    
    # Create summary
    create_summary(ts_datasets, nts_datasets)
    
    print("\n" + "="*70)
    print("Dataset download complete!")
    print("="*70)
    print("\nNext step: Run benchmark with real data")
    print("  python examples/04_simple_kmeans_benchmark.py")


if __name__ == '__main__':
    main()
