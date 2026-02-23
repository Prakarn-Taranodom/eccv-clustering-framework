# ECCV Algorithms - Implementation Guide

## 📍 ตำแหน่ง Algorithms

**Algorithms ทั้งหมดอยู่ที่:**
```
src/eccv/clustering/algorithms.py
```

## 🔍 Algorithms จาก Notebooks

### จาก `A_Benchmark_study_Clustering.ipynb`

Notebook นี้ใช้ algorithms หลักๆ ดังนี้:

#### 1. **Standard Clustering (sklearn)**
- ✅ `KMeans` → `'kmeans'`
- ✅ `SpectralClustering` → `'spectral'`
- ✅ `AgglomerativeClustering` → `'agglomerative'`
- ✅ `DBSCAN` → `'dbscan'`

#### 2. **Time Series Clustering (tslearn)**
- ✅ `TimeSeriesKMeans` → `'ts_kmeans'`
- ✅ `KShape` → `'kshape'`
- ✅ `KernelKMeans` → `'kernel_kmeans'`

#### 3. **Advanced TS Clustering (aeon)**
- ✅ `TimeSeriesCLARA` → `'clara'`
- ✅ `TimeSeriesCLARANS` → `'clarans'`

### จาก `A_Benchmark_study_ARIMA_GARCH.ipynb`

Notebook นี้ใช้สำหรับ:
- ✅ ARIMA modeling → `src/eccv/modeling/volatility.py`
- ✅ GARCH modeling → `src/eccv/modeling/volatility.py`
- ✅ CV extraction → `extract_cv_features()`

## 📦 การใช้งาน

### วิธีที่ 1: ใช้ผ่าน ClusteringPipeline

```python
from eccv.clustering.algorithms import ClusteringPipeline

# KMeans
pipeline = ClusteringPipeline('kmeans', n_clusters=3)
labels = pipeline.fit_predict(X)

# Spectral Clustering
pipeline = ClusteringPipeline('spectral', n_clusters=3)
labels = pipeline.fit_predict(X)

# Time Series KMeans
pipeline = ClusteringPipeline('ts_kmeans', n_clusters=3, metric='dtw')
labels = pipeline.fit_predict(X)
```

### วิธีที่ 2: ใช้ตรงๆ

```python
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans

# Standard KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Time Series KMeans
ts_kmeans = TimeSeriesKMeans(n_clusters=3, metric='dtw')
labels = ts_kmeans.fit_predict(X)
```

## 🔧 Algorithms ที่รองรับ

### ✅ พร้อมใช้งาน (ไม่ต้องติดตั้งเพิ่ม)

```python
ALGORITHMS = {
    'kmeans': KMeans,              # sklearn
    'spectral': SpectralClustering, # sklearn
    'agglomerative': AgglomerativeClustering, # sklearn
    'dbscan': DBSCAN,              # sklearn
}
```

### ⚙️ ต้องติดตั้ง tslearn

```bash
pip install tslearn
```

```python
ALGORITHMS.update({
    'ts_kmeans': TimeSeriesKMeans,
    'kshape': KShape,
    'kernel_kmeans': KernelKMeans,
})
```

### ⚙️ ต้องติดตั้ง aeon

```bash
pip install aeon
```

```python
ALGORITHMS.update({
    'clara': TimeSeriesCLARA,
    'clarans': TimeSeriesCLARANS,
})
```

## 📊 Benchmark Configuration

ใน `config/config.py`:

```python
CLUSTERING_CONFIG = {
    'algorithms': [
        'kmeans',           # ✅ พร้อมใช้
        'spectral',         # ✅ พร้อมใช้
        'agglomerative',    # ✅ พร้อมใช้
        'dbscan',           # ✅ พร้อมใช้
        'ts_kmeans',        # ⚙️ ต้องมี tslearn
        'kshape',           # ⚙️ ต้องมี tslearn
        'kernel_kmeans',    # ⚙️ ต้องมี tslearn
        'clara',            # ⚙️ ต้องมี aeon
        'clarans',          # ⚙️ ต้องมี aeon
    ],
}
```

## 🎯 ตัวอย่างจาก Notebooks

### Notebook: Clustering Workflow

```python
# 1. Load data
X, y = load_data()

# 2. Extract CV features
cv_features = extract_cv_features(X)

# 3. Cluster with raw features (baseline)
kmeans_raw = KMeans(n_clusters=3)
labels_raw = kmeans_raw.fit_predict(X)

# 4. Cluster with CV features
kmeans_cv = KMeans(n_clusters=3)
labels_cv = kmeans_cv.fit_predict(cv_features)

# 5. Compare
from sklearn.metrics import normalized_mutual_info_score
nmi_raw = normalized_mutual_info_score(y, labels_raw)
nmi_cv = normalized_mutual_info_score(y, labels_cv)
nmi_diff = nmi_cv - nmi_raw  # Improvement
```

### ใน Framework (เหมือนกัน แต่สะอาดกว่า)

```python
from eccv import DataLoader, extract_cv_features, ClusteringPipeline, ClusteringEvaluator

# 1. Load data
loader = DataLoader()
X, y = loader.load_ucr_dataset('GunPoint')

# 2. Extract CV
cv = extract_cv_features(X)

# 3. Cluster both
labels_raw = ClusteringPipeline('kmeans', n_clusters=3).fit_predict(X)
labels_cv = ClusteringPipeline('kmeans', n_clusters=3).fit_predict(cv)

# 4. Compare
evaluator = ClusteringEvaluator()
comparison = evaluator.compare_methods(y, labels_raw, labels_cv, X, cv)
print(comparison['improvements']['nmi_diff'])
```

## 🔄 Mapping: Notebook → Framework

| Notebook Code | Framework Code |
|---------------|----------------|
| `KMeans(n_clusters=3).fit_predict(X)` | `ClusteringPipeline('kmeans', n_clusters=3).fit_predict(X)` |
| `SpectralClustering(n_clusters=3).fit_predict(X)` | `ClusteringPipeline('spectral', n_clusters=3).fit_predict(X)` |
| `TimeSeriesKMeans(n_clusters=3).fit_predict(X)` | `ClusteringPipeline('ts_kmeans', n_clusters=3).fit_predict(X)` |
| Manual ARIMA-GARCH code | `extract_cv_features(X)` |
| Manual evaluation code | `ClusteringEvaluator().evaluate(y, labels)` |

## 📝 สรุป

**Algorithms อยู่ที่:**
- `src/eccv/clustering/algorithms.py` - Clustering algorithms
- `src/eccv/modeling/volatility.py` - ARIMA-GARCH (CV extraction)
- `src/eccv/evaluation/metrics.py` - Evaluation metrics

**ใช้งานผ่าน:**
- `ClusteringPipeline` - Unified interface
- `extract_cv_features()` - CV extraction
- `ClusteringEvaluator` - Evaluation

**ไม่มี hardcode:**
- Configuration ใน `config/config.py`
- เลือก algorithms ได้ตามต้องการ
- Parameters ปรับได้ทั้งหมด

---

**🎯 ทุกอย่างจาก notebooks ถูก refactor แล้ว!**
