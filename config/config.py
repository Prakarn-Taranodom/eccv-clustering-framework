"""
Configuration file for ECCV experiments.
Modify these parameters instead of hardcoding values in notebooks.
"""

# Dataset Configuration
DATASET_CONFIG = {
    'data_dir': './datasets',
    'timeseries_dir': './datasets/timeseries',
    'non_timeseries_dir': './datasets/non_timeseries',
    'results_dir': './results',
}

# UCR Time Series Datasets
UCR_DATASETS = [
    'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME',
    'Car', 'Coffee', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200', 'ECGFiveDays',
    'FaceFour', 'GunPoint', 'Ham', 'Herring', 'Lightning2', 'Lightning7',
    'Meat', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW', 'MoteStrain', 'OliveOil', 'OSULeaf',
    'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'Rock',
    'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'ToeSegmentation1',
    'ToeSegmentation2', 'TwoLeadECG', 'UMD', 'Wine', 'Chinatown',
    'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend'
]

# ARIMA-GARCH Configuration
ARIMA_GARCH_CONFIG = {
    'arima_order': (1, 0, 1),  # (p, d, q)
    'garch_p': 1,
    'garch_q': 1,
    'max_iter': 1000,
}

# Clustering Configuration
CLUSTERING_CONFIG = {
    'algorithms': [
        'kmeans',
        'spectral',
        'agglomerative',
        'ts_kmeans',
        'kshape',
        'kernel_kmeans',
        'clara',
        'clarans',
        'dbscan'
    ],
    'n_clusters_range': [2, 3, 4, 5, 6],
    'normalize': True,
    'random_state': 42,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': [
        'rand_index',
        'adjusted_rand_index',
        'nmi',
        'homogeneity',
        'completeness',
        'v_measure',
        'silhouette'
    ],
    'save_results': True,
    'results_format': 'csv',  # 'csv' or 'json'
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'normalization_method': 'zscore',  # 'zscore', 'minmax', 'robust'
    'handle_missing': 'interpolate',  # 'interpolate', 'forward_fill', 'drop'
    'outlier_detection': True,
    'outlier_threshold': 3.0,  # z-score threshold
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'n_runs': 10,  # Number of runs for each experiment
    'test_size': 0.3,
    'cross_validation': False,
    'cv_folds': 5,
    'parallel': False,
    'n_jobs': -1,
    'verbose': True,
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'save_plots': True,
    'plot_format': 'png',  # 'png', 'pdf', 'svg'
    'color_palette': 'Set2',
}
