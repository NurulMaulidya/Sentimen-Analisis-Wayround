# LAPORAN AUDIT EXPERIMEN

## Perbandingan Eksperimen Saat Ini vs Yang Diperlukan

---

## Ringkasan

| # | Eksperimen | Required | Current | Status |
|---|------------|----------|----------|---------|
| 0 | Class Distribution | ✅ WAJIB | ❌ Tidak Ada | **HARUS DITAMBAHKAN** |
| 1 | Split Ratio | ✅ | ✅ | SESUAI |
| 2 | Neutral Threshold | ✅ | ⚠️ | **PERBAIKAN** |
| 3 | TF-IDF Optimization | ✅ | ⚠️ | **PERBAIKAN** |
| 4 | SVM Kernel | ✅ | ⚠️ | **PERBAIKAN** |
| 5 | Class Imbalance | ✅ | ⚠️ | **PERBAIKAN** |
| 6 | Stratified CV | ✅ | ✅ | SESUAI |
| 7 | Overfitting | ✅ | ❌ Tidak Ada | **HARUS DITAMBAHKAN** |
| 8 | Feature Size vs Performance | ✅ | ❌ Tidak Ada | **HARUS DITAMBAHKAN** |
| 9 | Final Evaluation | ✅ | ✅ | SESUAI |

---

## Detail Perbedaan

### EXPERIMENT 0: Class Distribution (BARU - HARUS DITAMBAHKAN)

**Yang Diperlukan:**
```
- Jumlah Positif, Netral, Negatif
- Persentase masing-masing kelas
- Rasio kelas mayoritas : minoritas
- Visualisasi bar chart & pie chart
- Deteksi apakah dataset imbalanced
```

**Implementasi:**
```python
def run_experiment_0(df, text_column, label_column):
    dist = get_label_distribution(df, label_column)
    counts = {cls: dist[cls]['count'] for cls in SENTIMENT_CLASSES}
    total = sum(counts.values())
    majority = max(counts.values())
    minority = min(counts.values())
    ratio = majority / minority
    return {
        'total': total,
        'counts': counts,
        'percentages': {cls: dist[cls]['percentage'] for cls in SENTIMENT_CLASSES},
        'majority_class': max(counts, key=counts.get),
        'minority_class': min(counts, key=counts.get),
        'imbalance_ratio': ratio,
        'is_imbalanced': ratio > 2.0
    }
```

---

### EXPERIMENT 1: Split Ratio (SESUAI)

**Yang Diperlukan:**
```
- Ratio: 60:40, 70:30, 80:20, 90:10 ✅ SUDAH BENAR
```

**Current Implementation:** ✅ BENAR

---

### EXPERIMENT 2: Neutral Threshold (PERBAIKAN)

**Yang Diperlukan:**
```
T = 0.0, 0.1, 0.2, 0.3, 0.5, 1.0
```

**Current Implementation:**
```python
thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]  # ❌ SALAH
```

**Yang Seharusnya:**
```python
thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]  # ✅ BENAR
```

**FIX:** Ubah baris 144 di run_experiments.py dari `thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]` menjadi `thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]`

---

### EXPERIMENT 3: TF-IDF Optimization (PERBAIKAN)

**Yang Diperlukan:**
```
max_features = 1000, 2000, 3000, 5000
ngram_range = (1,1), (1,2), (1,3)
sublinear_tf = False, True
```

**Current Implementation:**
```python
tfidf_configs = [
    {"max_features": 1000, "ngram_range": (1, 1), ...},
    {"max_features": 3000, "ngram_range": (1, 1), ...},  # ❌ tidak ada 2000
    {"max_features": 5000, "ngram_range": (1, 1), ...},
    {"max_features": 10000, "ngram_range": (1, 1), ...},  # ❌ tidak perlu 10000
    ...
]
```

**Yang Seharusnya:**
```python
tfidf_configs = [
    # max_features = 1000
    {"max_features": 1000, "ngram_range": (1, 1), "sublinear_tf": True},
    {"max_features": 1000, "ngram_range": (1, 2), "sublinear_tf": True},
    {"max_features": 1000, "ngram_range": (1, 3), "sublinear_tf": True},
    {"max_features": 1000, "ngram_range": (1, 1), "sublinear_tf": False},
    {"max_features": 1000, "ngram_range": (1, 2), "sublinear_tf": False},
    # max_features = 2000
    {"max_features": 2000, "ngram_range": (1, 1), "sublinear_tf": True},
    {"max_features": 2000, "ngram_range": (1, 2), "sublinear_tf": True},
    {"max_features": 2000, "ngram_range": (1, 3), "sublinear_tf": True},
    {"max_features": 2000, "ngram_range": (1, 1), "sublinear_tf": False},
    {"max_features": 2000, "ngram_range": (1, 2), "sublinear_tf": False},
    # max_features = 3000
    {"max_features": 3000, "ngram_range": (1, 1), "sublinear_tf": True},
    {"max_features": 3000, "ngram_range": (1, 2), "sublinear_tf": True},
    {"max_features": 3000, "ngram_range": (1, 3), "sublinear_tf": True},
    # max_features = 5000
    {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
    {"max_features": 5000, "ngram_range": (1, 2), "sublinear_tf": True},
    {"max_features": 5000, "ngram_range": (1, 3), "sublinear_tf": True},
]
```

---

### EXPERIMENT 4: SVM Kernel & Hyperparameter (PERBAIKAN)

**Yang Diperlukan:**
```
kernel: linear, rbf, poly, sigmoid
C: 0.01, 0.1, 0.5, 1.0, 10
gamma: scale, auto, 0.01, 0.1, 1
```

**Current Implementation:**
```python
svm_configs = [
    {"kernel": "linear", "C": 0.01, "gamma": "scale"},
    {"kernel": "linear", "C": 0.1, "gamma": "scale"},
    {"kernel": "linear", "C": 1.0, "gamma": "scale"},
    {"kernel": "linear", "C": 10.0, "gamma": "scale"},
    {"kernel": "linear", "C": 100.0, "gamma": "scale"},  # ❌ tidak perlu 100
    ...
    # gamma 1 tidak diuji
]
```

**Yang Seharusnya:**
```python
svm_configs = [
    # Linear
    {"kernel": "linear", "C": 0.01},
    {"kernel": "linear", "C": 0.1},
    {"kernel": "linear", "C": 0.5},
    {"kernel": "linear", "C": 1.0},
    {"kernel": "linear", "C": 10},
    # RBF
    {"kernel": "rbf", "C": 0.01, "gamma": "scale"},
    {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
    {"kernel": "rbf", "C": 0.5, "gamma": "scale"},
    {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 10, "gamma": "scale"},
    {"kernel": "rbf", "C": 0.1, "gamma": "auto"},
    {"kernel": "rbf", "C": 1.0, "gamma": "auto"},
    {"kernel": "rbf", "C": 0.1, "gamma": 0.01},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.01},
    {"kernel": "rbf", "C": 10, "gamma": 0.01},
    {"kernel": "rbf", "C": 0.1, "gamma": 0.1},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.1},
    {"kernel": "rbf", "C": 10, "gamma": 0.1},
    {"kernel": "rbf", "C": 0.1, "gamma": 1},
    {"kernel": "rbf", "C": 1.0, "gamma": 1},
    {"kernel": "rbf", "C": 10, "gamma": 1},
    # Poly
    {"kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 2},
    {"kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 3},
    {"kernel": "poly", "C": 10, "gamma": "scale", "degree": 3},
    # Sigmoid
    {"kernel": "sigmoid", "C": 0.1, "gamma": "scale"},
    {"kernel": "sigmoid", "C": 1.0, "gamma": "scale"},
    {"kernel": "sigmoid", "C": 10, "gamma": "scale"},
]
```

---

### EXPERIMENT 5: Class Imbalance (PERBAIKAN)

**Yang Diperlukan:**
```
1. Tanpa balancing
2. Class weight
3. Random undersampling
4. Random oversampling
5. SMOTE
```

**Current Implementation:**
```python
weight_configs = [
    {"name": "None", "class_weight": None},           # ✅
    {"name": "balanced", "class_weight": "balanced"}, # ✅
    {"name": "Custom-1", "class_weight": {"Positif": 1.0, "Netral": 1.5, "Negatif": 2.0}},
    {"name": "Custom-2", "class_weight": {"Positif": 1.0, "Netral": 1.0, "Negatif": 3.0}},  # ✅
    {"name": "Custom-3", "class_weight": {"Positif": 1.0, "Netral": 2.0, "Negatif": 2.0}},
]
# ❌ Tidak ada undersampling, oversampling, SMOTE
```

**Yang Seharusnya:**
```python
# 1. No balancing
svm = SVC(kernel=kernel, C=C, class_weight=None)

# 2. Class weight balanced
svm = SVC(kernel=kernel, C=C, class_weight="balanced")

# 3. Custom class weight
svm = SVC(kernel=kernel, C=C, class_weight={"Positif": 1.0, "Netral": 1.0, "Negatif": 3.0})

# 4. Random Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# 5. Random Oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# 6. SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

### EXPERIMENT 6: Stratified CV (SESUAI ✅)

**Current Implementation:** BENAR
- 5-fold StratifiedKFold
- scoring: f1_macro

---

### EXPERIMENT 7: Overfitting Detection (BARU - HARUS DITAMBAHKAN)

**Yang Diperlukan:**
```
- Training Score
- Validation/CV Score
- Test Score
- Gap = F1_train - F1_test
- Rekomendasi berdasarkan gap
```

**Implementasi:**
```python
def run_experiment_7(df, text_column, label_column, split_ratio, tfidf_config, svm_config):
    X_train_text, X_test_text, y_train, y_test = train_test_split(...)
    X_train = vec.fit_transform(X_train_text)
    X_test = vec.transform(X_test_text)

    model.fit(X_train, y_train)
    train_f1 = f1_score(y_train, model.predict(X_train), average="macro")
    test_f1 = f1_score(y_test, model.predict(X_test), average="macro")
    gap = train_f1 - test_f1

    print(f"Train F1: {train_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Gap: {gap:.4f}")

    if gap > 0.15:
        print("[WARNING] HIGH OVERFITTING")
    elif gap > 0.10:
        print("[MODERATE] Some overfitting")
    else:
        print("[OK] No significant overfitting")

    return gap, {"train_f1": train_f1, "test_f1": test_f1, "gap": gap}
```

---

### EXPERIMENT 8: Feature Size vs Performance (BARU - HARUS DITAMBAHKAN)

**Yang Diperlukan:**
```
- max_features = 500, 1000, 2000, 3000, 5000
- ngram_range = (1,1), (1,2), (1,3)
- Bandingkan CV F1 score
- Analisis korelasi fitur vs performa
```

**Implementasi:**
```python
def run_experiment_8(df, text_column, label_column, split_ratio, svm_config):
    results = []
    for max_feat in [500, 1000, 2000, 3000, 5000]:
        vec = TfidfVectorizer(max_features=max_feat, ngram_range=(1,2), ...)
        X = vec.fit_transform(texts)
        scores = cross_val_score(SVC(**svm_config), X, y, cv=5, scoring="f1_macro")
        results.append({"max_features": max_feat, "cv_f1": scores.mean(), "n_features": X.shape[1]})
        print(f"max_features={max_feat} -> F1={scores.mean():.4f}")

    best = max(results, key=lambda x: x["cv_f1"])
    print(f"Best: max_features={best['max_features']}, F1={best['cv_f1']:.4f}")

    # Analisis korelasi
    corr, p_value = pearsonr([r['n_features'] for r in results], [r['cv_f1'] for r in results])
    print(f"Correlation: r={corr:.4f}, p={p_value:.4f}")

    return best["cv_f1"], {"best": best, "all_results": results}
```

---

### EXPERIMENT 9: Final Evaluation (SESUAI ✅)

**Current Implementation:** BENAR
- Accuracy, Precision, Recall, F1, Macro F1, Weighted F1
- Confusion Matrix
- Per-class metrics

---

## Checklist Perbaikan

### Wajib Diperbaiki:

- [ ] Tambahkan Experiment 0: Class Distribution
- [ ] Fix Experiment 2: thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
- [ ] Fix Experiment 3: Tambahkan max_features=2000, hapus max_features=10000
- [ ] Fix Experiment 4: C = [0.01, 0.1, 0.5, 1.0, 10] (hapus 100), tambahkan gamma=1
- [ ] Fix Experiment 5: Tambahkan undersampling, oversampling, SMOTE
- [ ] Tambahkan Experiment 7: Overfitting Detection
- [ ] Tambahkan Experiment 8: Feature Size vs Performance

### Verifikasi:

- [ ] File dapat di-import tanpa error
- [ ] Semua 10 eksperimen berjalan
- [ ] Output tersimpan dengan benar di best_config.json

---

## Catatan

File backup tersedia di: `scripts/run_experiments/run_experiments_backup.py`

Untuk memperbaiki file secara manual:
1. Backup file saat ini
2. Edit bagian yang perlu diperbaiki
3. Verifikasi dengan menjalankan `python -c "import ast; ast.parse(open('run_experiments.py').read())"`

