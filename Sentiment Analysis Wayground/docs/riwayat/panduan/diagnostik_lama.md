# Troubleshooting Guide
## Analisis Sentimen Wayground

---

## Daftar Isi

1. [Instalasi](#1-instalasi)
2. [Pipeline](#2-pipeline)
3. [Eksperimen](#3-eksperimen)
4. [Model](#4-model)
5. [Data](#5-data)

---

## 1. Instalasi

### ❌ Error: `ModuleNotFoundError: No module named 'sklearn'`

**Penyebab:** scikit-learn belum terinstall

**Solusi:**
```bash
pip install scikit-learn
```

### ❌ Error: `ModuleNotFoundError: No module named 'Sastrawi'`

**Penyebab:** Library Sastrawi belum terinstall

**Solusi:**
```bash
pip install Sastrawi
```

### ❌ Error: `ModuleNotFoundError: No module named 'google_play_scraper'`

**Penyebab:** Google Play Scraper belum terinstall

**Solusi:**
```bash
pip install google-play-scraper
```

### ❌ Error: `PermissionError: [WinError 5] Access is denied`

**Penyebab:** Tidak punya izin tulis di folder

**Solusi:**
```bash
# Jalankan CMD/PowerShell sebagai Administrator
# Atau gunakan virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### ❌ Error: `Python version not supported`

**Penyebab:** Python versi terlalu lama

**Solusi:**
```bash
# Install Python 3.10 atau lebih baru
python --version  # Cek versi
```

---

## 2. Pipeline

### ❌ Error: `FileNotFoundError: [Errno 2] No such file or directory: 'data/...`

**Penyebab:** File data tidak ditemukan

**Solusi:**
```bash
# Pastikan pipeline sudah dijalankan dari awal
python pipeline/run_pipeline.py

# Atau scrap data terlebih dahulu
python pipeline/scraper.py
```

### ❌ Error: `KeyError: 'processed_text'`

**Penyebab:** Kolom processed_text tidak ada di DataFrame

**Solusi:**
```bash
# Jalankan preprocessing terlebih dahulu
python pipeline/run_pipeline.py --stage 2
```

### ❌ Error: `UnicodeDecodeError`

**Penyebab:** File encoding tidak kompatibel

**Solusi:**
```python
# Gunakan encoding utf-8
df = pd.read_csv('file.csv', encoding='utf-8-sig')
```

### ⚠️ Warning: `Empty DataFrame`

**Penyebab:** Semua data dihapus saat preprocessing

**Solusi:**
```python
# Cek data asli
df = pd.read_csv('data/02_data_akhir.csv')
print(f"Jumlah data: {len(df)}")

# Cek kolom
print(df.columns)
```

### ⚠️ Warning: `Low preprocessing retention`

**Penyebab:** Banyak teks kosong setelah preprocessing

**Solusi:**
```python
# Cek contoh teks yang dihapus
df_raw = pd.read_csv('data/02_data_akhir.csv')
df_clean = pd.read_csv('data/03_cleaned_data.csv')

print(f"Raw: {len(df_raw)}, Clean: {len(df_clean)}")

# Contoh teks yang dihapus
removed = df_raw[~df_raw.index.isin(df_clean.index)]
print(removed['content'].head())
```

---

## 3. Eksperimen

### ❌ Error: `ValueError: Unknown label type`

**Penyebab:** Label tidak sesuai format

**Solusi:**
```python
# Pastikan label sesuai
print(df['sentiment_label'].unique())
# Output: ['Positif', 'Netral', 'Negatif']
```

### ❌ Error: `ValueError: Found input variables with inconsistent numbers of samples`

**Penyebab:** X dan y tidak sama panjang

**Solusi:**
```python
# Cek panjang data
print(f"X: {len(X)}, y: {len(y)}")

# Filter data kosong
mask = df['processed_text'].notna() & (df['processed_text'] != '')
df_filtered = df[mask]
```

### ⚠️ Warning: `CV F1 Score sangat rendah`

**Penyebab:** Kemungkinan besar:
1. Dataset tidak seimbang
2. Parameter tidak optimal
3. Preprocessing terlalu agresif

**Solusi:**
```python
# 1. Cek distribusi kelas
print(df['sentiment_label'].value_counts())

# 2. Gunakan class_weight
model = SVC(class_weight='balanced')

# 3. Jalankan eksperimen ulang
python scripts/run_experiments/run_experiments.py
```

### ⚠️ Warning: `Stabilitas rendah (std > 0.05)`

**Penyebab:** Model tidak stabil antar fold

**Solusi:**
```python
# Tambah data training
# Atau gunakan data augmentation
# Atau gunakan model yang lebih sederhana
```

---

## 4. Model

### ❌ Error: `NotFittedError: This TextVectorizer instance is not fitted yet`

**Penyebab:** Vectorizer belum di-fit

**Solusi:**
```python
# Fit vectorizer terlebih dahulu
vectorizer = TextVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
```

### ❌ Error: `AttributeError: 'SVC' object has no attribute 'predict_proba'`

**Penyebab:** SVC dengan kernel tertentu tidak mendukung predict_proba

**Solusi:**
```python
# Gunakan decision_function sebagai confidence
distances = model.decision_function(X_test)
confidence = np.max(np.abs(distances), axis=1)

# Atau gunakan CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(model)
calibrated_model.fit(X_train, y_train)
proba = calibrated_model.predict_proba(X_test)
```

### ⚠️ Warning: `Overfitting detected`

```
Training F1: 0.99
Test F1: 0.85
Gap: 0.14
```

**Penyebab:**
1. Model terlalu kompleks
2. Dataset terlalu kecil
3. Fitur terlalu banyak

**Solusi:**
```python
# 1. Kurangi regularisasi (C lebih besar)
model = SVC(C=100)

# 2. Kurangi max_features
vectorizer = TfidfVectorizer(max_features=1000)

# 3. Tambah data training

# 4. Gunakan dropout/features selection
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=500)
X_selected = selector.fit_transform(X_train, y_train)
```

### ⚠️ Warning: `Underfitting detected`

```
Training F1: 0.65
Test F1: 0.63
```

**Penyebab:**
1. Model terlalu sederhana
2. Fitur tidak informatif
3. Dataset tidak relevan

**Solusi:**
```python
# 1. Tambah regularisasi (C lebih kecil)
model = SVC(C=0.1)

# 2. Tambah fitur (max_features lebih besar)
vectorizer = TfidfVectorizer(max_features=10000)

# 3. Gunakan kernel yang lebih fleksibel
model = SVC(kernel='rbf', C=10, gamma='scale')
```

---

## 5. Data

### ❌ Error: `No reviews found during scraping`

**Penyebab:**
1. Koneksi internet bermasalah
2. Google Play rate limiting
3. Package name salah

**Solusi:**
```python
# 1. Cek package name
PACKAGE_NAME = "com.quizizz_mobile"

# 2. Tambah delay
REQUEST_DELAY = 2.0  # seconds

# 3. Coba lagi nanti
```

### ❌ Error: `Invalid date range`

**Penyebab:** Tanggal tidak sesuai format

**Solusi:**
```python
from datetime import datetime
import pandas as pd

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2026-07-01")
```

### ⚠️ Warning: `Imbalanced dataset`

```
Positif: 1500 (60%)
Netral: 800 (32%)
Negatif: 200 (8%)
```

**Penyebab:** Dataset tidak seimbang secara alami

**Solusi:**
```python
# 1. Class Weight
model = SVC(class_weight={'Negatif': 3.0})

# 2. Oversampling (SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 3. Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
```

### ⚠️ Warning: `Low inter-rater agreement`

**Penyebab:** Pelabelan lexicon tidak akurat

**Solusi:**
```python
# 1. Tune threshold
for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
    results = df['processed_text'].apply(
        lambda x: classify_sentiment(x, threshold=t)
    )
    # Hitung agreement dengan ground truth

# 2. Validasi manual
# Sample 100 data, bandingkan dengan pelabelan manual
```

---

## Checklist Debugging

Jika terjadi error, cek checklist berikut:

- [ ] Python version >= 3.10
- [ ] Semua dependencies terinstall
- [ ] Data file ada di lokasi yang benar
- [ ] Konfigurasi sesuai (best_config.json)
- [ ] Virtual environment activated
- [ ] Permissions cukup (bisa tulis file)

---

## Getting Help

```bash
# Cek versi library
python -c "import sklearn; print(sklearn.__version__)"
python -c "import pandas; print(pandas.__version__)"

# Cek path
python -c "import sys; print(sys.path)"

# Cek file exists
import os
print(os.path.exists('data/pipeline_output/stage3_labeled.csv'))
```

---

## Log Error

Jika masih ada error, catat:

1. **Error message lengkap:**
```
Traceback (most recent call last):
  File "pipeline/run_pipeline.py", line 123, in <module>
    ...
ValueError: Invalid input
```

2. **Data yang digunakan:**
```
File: data/03_cleaned_data.csv
Rows: 3141
Columns: ['content', 'rating', ...]
```

3. **Konfigurasi:**
```json
{
  "svm_C": 10.0,
  "svm_kernel": "linear"
}
```

4. **Environment:**
```
Python: 3.10.0
OS: Windows 10
pandas: 2.0.0
sklearn: 1.3.0
```
