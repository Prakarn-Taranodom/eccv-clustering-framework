"""
Example: Basic ECCV workflow
This script demonstrates how to use the ECCV framework for clustering.
"""

import sys
sys.path.append('./src')

import numpy as np
from eccv.utils.data_loader import DataLoader
from eccv.modeling.volatility import extract_cv_features
from eccv.clustering.algorithms import ClusteringPipeline
from eccv.evaluation.metrics import ClusteringEvaluator

# Import configuration
sys.path.append('./config')
from config import ARIMA_GARCH_CONFIG, CLUSTERING_CONFIG


def main():
    """Run basic ECCV clustering example."""
    
    print("="*60)
    print("ECCV: Enhanced Clustering using Conditional Volatility")
    print("="*60)
    
    # Step 1: Load Data
    print("\n[1/5] Loading dataset...")
    loader = DataLoader()
    
    # Load a sample UCR dataset
    dataset_name = 'GunPoint'
    X, y = loader.load_ucr_dataset(dataset_name)
    print(f"   Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[-1]} timepoints")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Step 2: Extract Conditional Volatility Features
    print("\n[2/5] Extracting conditional volatility features...")
    cv_features = extract_cv_features(
        X,
        arima_order=ARIMA_GARCH_CONFIG['arima_order'],
        garch_p=ARIMA_GARCH_CONFIG['garch_p'],
        garch_q=ARIMA_GARCH_CONFIG['garch_q']
    )
    print(f"   CV features shape: {cv_features.shape}")
    
    # Step 3: Clustering with Original Features (Baseline)
    print("\n[3/5] Clustering with original features (baseline)...")
    n_clusters = len(np.unique(y))
    
    baseline_pipeline = ClusteringPipeline(
        algorithm='kmeans',
        n_clusters=n_clusters,
        normalize=True,
        random_state=42
    )
    y_pred_baseline = baseline_pipeline.fit_predict(X)
    print(f"   Baseline clustering completed")
    
    # Step 4: Clustering with CV Features
    print("\n[4/5] Clustering with CV features...")
    cv_pipeline = ClusteringPipeline(
        algorithm='kmeans',
        n_clusters=n_clusters,
        normalize=True,
        random_state=42
    )
    y_pred_cv = cv_pipeline.fit_predict(cv_features)
    print(f"   CV-enhanced clustering completed")
    
    # Step 5: Evaluate and Compare
    print("\n[5/5] Evaluating results...")
    evaluator = ClusteringEvaluator()
    
    comparison = evaluator.compare_methods(
        y_true=y,
        y_pred_baseline=y_pred_baseline,
        y_pred_cv=y_pred_cv,
        X_baseline=X,
        X_cv=cv_features
    )
    
    # Print results
    evaluator.print_results(comparison['baseline'], "Baseline (Original Features)")
    evaluator.print_results(comparison['cv_enhanced'], "CV-Enhanced")
    evaluator.print_results(comparison['improvements'], "Improvements")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    nmi_improvement = comparison['improvements'].get('nmi_diff', 0)
    ri_improvement = comparison['improvements'].get('rand_index_diff', 0)
    
    if nmi_improvement > 0:
        print(f"✓ NMI improved by {nmi_improvement:.4f}")
    else:
        print(f"✗ NMI decreased by {abs(nmi_improvement):.4f}")
    
    if ri_improvement > 0:
        print(f"✓ Rand Index improved by {ri_improvement:.4f}")
    else:
        print(f"✗ Rand Index decreased by {abs(ri_improvement):.4f}")
    
    print("="*60)


if __name__ == "__main__":
    main()
