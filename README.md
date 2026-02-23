# ECCV: Enhanced Clustering using Conditional Volatility

[![Framework](https://img.shields.io/badge/Framework-ECCV-blue)](https://github.com/Prakarn-Taranodom/eccv-clustering-framework)
[![Implementation](https://img.shields.io/badge/Implementation-Available-green)](https://github.com/Prakarn-Taranodom/eccv-implementation)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A novel framework for time-series clustering using conditional volatility features extracted through ARIMA-GARCH modeling.**

> 📌 **Note:** This repository contains the **conceptual framework and core implementation**. For complete benchmark studies, real-world applications, and detailed experimental results, visit the [**Implementation Repository**](https://github.com/Prakarn-Taranodom/eccv-implementation).

---

## 🎯 Overview

ECCV (Enhanced Clustering using Conditional Volatility) transforms time-series clustering by extracting **conditional volatility patterns** as discriminative features for clustering algorithms.

### Core Concept

Traditional clustering uses raw time-series values. ECCV extracts **volatility patterns** that may reveal hidden structures:

```
Raw Time Series → ARIMA (Remove Trend) → GARCH (Extract Volatility) → CV Features → Clustering
```

![ECCV Framework](images/ECCV_framework_diagram.png)

---

## 🔬 Methodology

### 1. Conditional Volatility Extraction

**ARIMA-GARCH Pipeline:**
- **ARIMA(p,d,q)**: Removes trend and seasonal components
- **GARCH(p,q)**: Models volatility clustering in residuals  
- **Output**: Conditional volatility time series as features

### 2. Preprocessing Considerations

⚠️ **Important:** Performance depends heavily on preprocessing choices:

**Before ARIMA-GARCH:**
- Stationarity testing (ADF, KPSS)
- Outlier detection and handling
- Missing value imputation

**Before Clustering:**
- Feature scaling methods:
  - Z-score standardization
  - Min-Max normalization
  - Yeo-Johnson transformation
  - RobustScaler
- Choice depends on data distribution and clustering algorithm

**Note:** Different preprocessing strategies yield different results. The effectiveness of CV features varies by:
- Data characteristics (volatility patterns, stationarity)
- Preprocessing methods (scaling, transformation)
- Clustering algorithm choice
- ARIMA-GARCH parameter selection

---

## 📊 When to Use ECCV

### ✅ Suitable For:
- **Financial time series** (stock prices, returns, forex)
- **Data with volatility clustering** (GARCH effects)
- **Economic indicators** with changing variance
- **Sensor data** with heteroscedastic noise

### ❌ May Not Help:
- **Non-time-series data** (tabular, cross-sectional)
- **Stationary data** without volatility patterns
- **Small datasets** (insufficient for GARCH estimation)
- **Data without temporal dependencies**

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Prakarn-Taranodom/eccv-clustering-framework.git
cd eccv-clustering-framework
pip install -r requirements.txt
```

### Basic Usage

```python
from eccv.modeling.volatility import extract_cv_features
from eccv.clustering.algorithms import ClusteringPipeline

# Extract conditional volatility features
cv_features = extract_cv_features(
    X, 
    arima_order=(1, 0, 1),
    garch_p=1, 
    garch_q=1
)

# Cluster with CV features (Yeo-Johnson transformation applied)
pipeline = ClusteringPipeline(
    algorithm='kmeans',
    n_clusters=3,
    standardize=True  # Uses Yeo-Johnson + standardization
)
labels = pipeline.fit_predict(cv_features)
```

### Run Examples

```bash
# Download real datasets
python examples/06_download_real_datasets.py

# Basic demonstration
python examples/01_basic_example.py
```

---

## 📁 Repository Structure

```
eccv-clustering-framework/
├── src/eccv/              # Core framework
│   ├── modeling/          # ARIMA-GARCH volatility extraction
│   ├── clustering/        # Clustering algorithms wrapper
│   ├── evaluation/        # Performance metrics
│   └── utils/             # Data loading utilities
├── examples/              # Simple usage examples
├── datasets/              # Sample datasets
├── images/                # Framework diagrams
├── config/                # Configuration files
└── docs/                  # Documentation
```

---

## 🔗 Full Implementation & Benchmarks

For comprehensive studies and real-world applications:

### 👉 [ECCV Implementation Repository](https://github.com/Prakarn-Taranodom/eccv-implementation)

**Includes:**
- ✅ **Extensive benchmark studies** (40+ datasets)
- ✅ **Real-world Thai stock market analysis**
- ✅ **Statistical significance tests**
- ✅ **Comprehensive evaluation results**
- ✅ **Jupyter notebooks** with detailed analysis
- ✅ **Publication-ready figures and tables**
- ✅ **Parameter optimization studies**
- ✅ **Preprocessing comparison experiments**

---

## 📈 Key Insights

### Performance Factors

ECCV effectiveness depends on multiple factors:

1. **Data Characteristics**
   - Presence of volatility clustering
   - Stationarity properties
   - Sample size (GARCH requires sufficient data)

2. **Preprocessing Choices**
   - Scaling method (standardization, normalization, transformation)
   - Outlier handling strategy
   - Stationarity transformation

3. **Model Parameters**
   - ARIMA order (p, d, q)
   - GARCH specification (p, q)
   - Clustering algorithm selection

4. **Evaluation Metrics**
   - Results vary by metric (RI, NMI, ARI, Silhouette)
   - Different metrics may show different trends

### Important Notes

⚠️ **CV features are NOT universally beneficial:**
- Performance improvements are **dataset-specific**
- Proper **preprocessing is critical**
- **Parameter tuning** may be necessary
- Results should be **validated** on domain-specific data

---

## 🛠️ Technologies

- **Python 3.8+**
- **Time Series Modeling:** `statsmodels`, `arch`
- **Clustering:** `scikit-learn`, `tslearn`
- **Data Processing:** `pandas`, `numpy`
- **Preprocessing:** `sklearn.preprocessing.PowerTransformer` (Yeo-Johnson)

---

## 📚 Documentation

- [ALGORITHMS_GUIDE.md](ALGORITHMS_GUIDE.md) - Clustering algorithms overview
- [Implementation Repo](https://github.com/Prakarn-Taranodom/eccv-implementation) - Full benchmarks and analysis

---

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{taranodom2024eccv,
  author = {Taranodom, Prakarn},
  title = {ECCV: Enhanced Clustering using Conditional Volatility},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Prakarn-Taranodom/eccv-clustering-framework}}
}
```

---

## 👤 Author

**Prakarn Taranodom**

- GitHub: [@Prakarn-Taranodom](https://github.com/Prakarn-Taranodom)
- Framework: [eccv-clustering-framework](https://github.com/Prakarn-Taranodom/eccv-clustering-framework)
- Implementation: [eccv-implementation](https://github.com/Prakarn-Taranodom/eccv-implementation)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⭐ Acknowledgments

- UCR Time Series Archive for benchmark datasets
- Thai Stock Exchange for real-world data (in implementation repo)
- Open-source community for excellent libraries (`statsmodels`, `arch`, `scikit-learn`)

---

## 🔍 Related Work

- **Implementation Repository:** [eccv-implementation](https://github.com/Prakarn-Taranodom/eccv-implementation) - Complete benchmark studies and real-world applications
- **Research Paper:** [Link to paper when published]
- **Thesis:** This framework is part of a Master's thesis research

---

**⚡ Quick Links:**
- [Framework Repository](https://github.com/Prakarn-Taranodom/eccv-clustering-framework) (You are here)
- [Implementation & Benchmarks](https://github.com/Prakarn-Taranodom/eccv-implementation)
- [Issues](https://github.com/Prakarn-Taranodom/eccv-clustering-framework/issues)
