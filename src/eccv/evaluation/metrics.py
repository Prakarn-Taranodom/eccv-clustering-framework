"""Evaluation metrics for clustering performance."""

import numpy as np
from sklearn.metrics import (
    rand_score, 
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score
)
from typing import Dict, Optional


class ClusteringEvaluator:
    """Evaluate clustering performance with multiple metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def evaluate(self, 
                y_true: np.ndarray, 
                y_pred: np.ndarray,
                X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate clustering results.
        
        Args:
            y_true: True labels
            y_pred: Predicted cluster labels
            X: Original data (optional, for silhouette score)
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # External metrics (require true labels)
        results['rand_index'] = rand_score(y_true, y_pred)
        results['adjusted_rand_index'] = adjusted_rand_score(y_true, y_pred)
        results['nmi'] = normalized_mutual_info_score(y_true, y_pred)
        results['homogeneity'] = homogeneity_score(y_true, y_pred)
        results['completeness'] = completeness_score(y_true, y_pred)
        results['v_measure'] = v_measure_score(y_true, y_pred)
        
        # Internal metrics (don't require true labels)
        if X is not None:
            try:
                # Flatten if 3D
                if X.ndim == 3:
                    X_flat = X.reshape(X.shape[0], -1)
                else:
                    X_flat = X
                
                results['silhouette'] = silhouette_score(X_flat, y_pred)
            except Exception as e:
                print(f"Could not compute silhouette score: {e}")
                results['silhouette'] = np.nan
        
        self.metrics = results
        return results
    
    def compare_methods(self,
                       y_true: np.ndarray,
                       y_pred_baseline: np.ndarray,
                       y_pred_cv: np.ndarray,
                       X_baseline: Optional[np.ndarray] = None,
                       X_cv: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare baseline vs CV-enhanced clustering.
        
        Args:
            y_true: True labels
            y_pred_baseline: Predictions without CV
            y_pred_cv: Predictions with CV
            X_baseline: Original features
            X_cv: CV features
            
        Returns:
            Dictionary with results for both methods and differences
        """
        baseline_results = self.evaluate(y_true, y_pred_baseline, X_baseline)
        cv_results = self.evaluate(y_true, y_pred_cv, X_cv)
        
        # Calculate improvements
        improvements = {}
        for metric in baseline_results:
            if not np.isnan(baseline_results[metric]) and not np.isnan(cv_results[metric]):
                improvements[f'{metric}_diff'] = cv_results[metric] - baseline_results[metric]
                improvements[f'{metric}_improvement_%'] = (
                    (cv_results[metric] - baseline_results[metric]) / 
                    max(abs(baseline_results[metric]), 1e-10) * 100
                )
        
        return {
            'baseline': baseline_results,
            'cv_enhanced': cv_results,
            'improvements': improvements
        }
    
    def print_results(self, results: Dict[str, float], title: str = "Results"):
        """Pretty print evaluation results."""
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        for metric, value in results.items():
            print(f"{metric:30s}: {value:8.4f}")
        print(f"{'='*50}\n")


def evaluate_clustering(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       X: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Convenience function for quick evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        X: Data (optional)
        
    Returns:
        Dictionary of metrics
    """
    evaluator = ClusteringEvaluator()
    return evaluator.evaluate(y_true, y_pred, X)
