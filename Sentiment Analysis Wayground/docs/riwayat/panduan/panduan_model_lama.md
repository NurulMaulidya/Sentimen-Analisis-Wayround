# Buku Panduan Model Analisis Sentimen Wayground

## Panduan Lengkap Menjalankan dan Mengoptimisasi Model SVM

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Struktur Project](#2-struktur-project)
3. [Instalasi](#3-instalasi)
4. [Menjalankan Pipeline](#4-menjalankan-pipeline)
5. [Eksperimen Optimisasi](#5-eksperimen-optimisasi)
6. [Konfigurasi Model](#6-konfigurasi-model)
7. [Pipeline Details](#7-pipeline-details)
8. [Troubleshooting](#8-troubleshooting)
9. [Referensi](#9-referensi)

---

## 1. Pendahuluan

### 1.1 Tentang Project

Project ini adalah implementasi analisis sentimen ulasan Google Play Store aplikasi **Wayground (Quizizz)** menggunakan metode **Support Vector Machine (SVM)** dengan pembobotan **TF-IDF**.

### 1.2 Fitur Utama

- **Scraping Data**: Mengambil ulasan dari Google Play Store
- **Preprocessing 7 Tahap**: Cleaning, case folding, tokenizing, normalisasi, negation handling, stopword removal, stemming
- **Pelabelan Otomatis**: Menggunakan InSet Lexicon
- **Deteksi Aspek**: 4 aspek pembelajaran (Kemudahan Akses, Gamifikasi, Materi/Konten, Performa Teknis)
- **Klasifikasi SVM**: Dengan GridSearchCV untuk optimasi hyperparameter
- **Evaluasi**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 1.3 Requirements

```
Python >= 3.10
pandas
numpy
scikit-learn
Sastrawi
google-play-scraper
tqdm
matplotlib
```

---

## 2. Struktur Project

```
Sentiment Analysis Wayground/
├── backend/                    # Flask API
├── data/
│   ├── lexicon/               # Kamus lexicon
│   │   ├── lexicon_positive.tsv   # Kata positif (192 kata)
│   │   └── lexicon_negative.tsv   # Kata negatif (111 kata)
│   ├── pipeline_output/       # Output pipeline
│   └── uploads/              # User uploads
├── docs/                      # Dokumentasi
├── models/                    # Model tersimpan
├── output/
│   ├── experiments/
│   │   └── best_config.json   # ✅ Single Source of Truth
│   └── evaluation/            # Hasil evaluasi
├── pipeline/                  # Pipeline utama
│   ├── scraper.py            # Scraping Google Play
│   ├── labeling.py          # Pelabelan lexicon
│   ├── aspect_detection.py  # Deteksi aspek
│   ├── train_svm.py        # Training SVM
│   └── run_pipeline.py     # Pipeline otomatis
├── scripts/
│   ├── config_loader.py     # Load konfigurasi
│   └── run_experiments/
│       └── run_experiments.py  # Jalankan semua eksperimen
├── shared/                    # Modul bersama
│   ├── constants.py        # Konstanta
│   ├── final_config.py     # Konfigurasi ML
│   ├── lexicon.py          # Modul lexicon
│   ├── preprocessing.py    # Preprocessing teks
│   └── svm_visualization.py # Visualisasi
├── tests/                    # Unit tests
└── _archive/                # File lama (arsip)
```

---

## 3. Instalasi

### 3.1 Clone/Download Project

```bash
# Clone repository atau download project
cd "Sentiment Analysis Wayground"
```

### 3.2 Buat Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3 Install Dependencies

```bash
pip install pandas numpy scikit-learn Sastrawi google-play-scraper tqdm matplotlib
```

### 3.4 Verifikasi Instalasi

```bash
python -c "import pandas, sklearn, Sastrawi; print('OK')"
```

---

## 4. Menjalankan Pipeline

### 4.1 Pipeline Otomatis (Recommended)

Jalankan seluruh pipeline dengan satu perintah:

```bash
cd Sentiment\ Analysis\ Wayground
python pipeline/run_pipeline.py
```

**Output:**
```
================================================================================
[PIPELINE] PIPELINE ANALISIS SENTIMEN WAYGROUND
================================================================================
Started at: 2024-01-15 10:30:00
Output directory: data/pipeline_output

[STAGE 1] SCRAPING DATA
...

[STAGE 2] PREPROCESSING
...

[STAGE 3] SENTIMENT LABELING
...

[STAGE 4] ASPECT DETECTION
...

[STAGE 5] DATA SPLITTING
...

[STAGE 6] SVM TRAINING
...

[OK] PIPELINE COMPLETED
```

### 4.2 Pipeline Per Tahap

#### Tahap 1: Scraping

```bash
python pipeline/scraper.py
```

**Output Files:**
- `data/01_semua_data.csv` - Data mentah
- `data/02_data_akhir.csv` - Data setelah filter
- `data/03_data_audit_scraping.csv` - Audit bahasa

#### Tahap 2: Preprocessing

```bash
python -c "
import pandas as pd
from shared.preprocessing import preprocess_text
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv('data/02_data_akhir.csv')
df['processed_text'] = df['content'].progress_apply(preprocess_text)
df.to_csv('data/03_cleaned_data.csv', index=False)
print(f'Processed {len(df)} reviews')
"
```

#### Tahap 3: Pelabelan

```bash
python -c "
import pandas as pd
from shared.lexicon import classify_sentiment

df = pd.read_csv('data/03_cleaned_data.csv')
results = df['processed_text'].apply(lambda x: classify_sentiment(x, threshold=1.0))
df['sentiment_label'] = results.apply(lambda x: x['label'])
df['sentiment_score'] = results.apply(lambda x: x['raw_score'])
df.to_csv('data/pipeline_output/stage3_labeled.csv', index=False)
"
```

#### Tahap 4: Deteksi Aspek

```bash
python pipeline/aspect_detection.py
```

#### Tahap 5-6: Split & Training

```bash
python pipeline/train_svm.py
```

### 4.3 Opsi Pipeline

```bash
# Resume dari tahap tertentu
python pipeline/run_pipeline.py --stage 3

# Scraping ulang
python pipeline/run_pipeline.py --rescrape

# Mulai dari labeling
python pipeline/run_pipeline.py --from-labeling

# Mulai dari training
python pipeline/run_pipeline.py --from-training
```

---

## 5. Eksperimen Optimisasi

### 5.1 Jalankan Semua Eksperimen

```bash
python scripts/run_experiments/run_experiments.py
```

**Ini akan menjalankan 6 eksperimen:**

| # | Eksperimen | Deskripsi |
|---|------------|-----------|
| 1 | Split Ratio | Optimasi 60:40, 70:30, 80:20, 90:10 |
| 2 | Neutral Threshold | Optimasi T=0.0, 0.5, 1.0, 1.5, 2.0 |
| 3 | TF-IDF | Optimasi max_features, ngram_range, sublinear_tf |
| 4 | SVM Kernel | Optimasi kernel, C, gamma |
| 5 | Class Imbalance | Testing class_weight |
| 6 | Stratified CV | Validasi stabilitas 5-fold |

### 5.2 Output Eksperimen

Setelah selesai, hasil tersimpan di:
```
output/experiments/best_config.json
```

**Contoh best_config.json:**
```json
{
  "split_ratio": 0.8,
  "neutral_threshold": 1.0,
  "tfidf_max_features": 5000,
  "tfidf_ngram_range": [1, 2],
  "tfidf_sublinear_tf": true,
  "svm_kernel": "linear",
  "svm_C": 10.0,
  "svm_gamma": "scale",
  "svm_class_weight": {"Negatif": 3.0}
}
```

### 5.3 Eksperimen Individual

#### Eksperimen 1: Split Ratio

```python
from sklearn.model_selection import train_test_split

split_ratios = [0.6, 0.7, 0.8, 0.9]  # 60:40, 70:30, 80:20, 90:10

for ratio in split_ratios:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1-ratio,
        random_state=42,
        stratify=y
    )
    # Evaluasi dan catat hasil
```

#### Eksperimen 2: Neutral Threshold

```python
from shared.lexicon import classify_sentiment

thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]

for t in thresholds:
    results = df['processed_text'].apply(
        lambda x: classify_sentiment(x, threshold=t)
    )
    # Hitung distribusi dan F1-score
```

#### Eksperimen 3: TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

configs = [
    {"max_features": 1000, "ngram_range": (1, 1)},
    {"max_features": 3000, "ngram_range": (1, 1)},
    {"max_features": 5000, "ngram_range": (1, 2)},
    {"max_features": 10000, "ngram_range": (1, 3)},
]

for config in configs:
    vec = TfidfVectorizer(**config)
    X_vec = vec.fit_transform(texts)
    # Evaluasi dan catat CV F1-score
```

#### Eksperimen 4: SVM Kernel

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01, 0.1, 0.5, 1.0, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

grid = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

#### Eksperimen 5: Class Imbalance

```python
weight_configs = [
    None,                                    # Tanpa balancing
    'balanced',                              # Balanced auto
    {"Positif": 1.0, "Netral": 1.0, "Negatif": 2.0},
    {"Positif": 1.0, "Netral": 1.0, "Negatif": 3.0},
]

for weight in weight_configs:
    model = SVC(kernel='linear', C=10, class_weight=weight)
    model.fit(X_train, y_train)
    # Evaluasi F1-score per kelas
```

### 5.4 Interpretasi Hasil

#### Memilih Parameter Terbaik

1. **CV F1-Score Tertinggi**: Pilih konfigurasi dengan F1-score rata-rata tertinggi
2. **Stabilitas**: Perhatikan standar deviasi (std < 0.02 = stabil)
3. **Generalization Gap**: Selisih F1 train vs test (ideal < 0.10)

#### Contoh Interpretasi

```
Experiment 4: SVM Kernel
=======================
kernel     C       gamma     CV F1 Mean
-------    ------  --------  -----------
linear     10.0    scale     0.8469
rbf        10.0    scale     0.8321
poly       3       scale     0.8156
sigmoid    10.0    scale     0.7989

Best: linear, C=10.0, gamma=scale (F1: 0.8469)
```

---

## 6. Konfigurasi Model

### 6.1 Single Source of Truth

**IMPORTANT:** Semua konfigurasi model harus berasal dari `output/experiments/best_config.json`

### 6.2 Konfigurasi Final (Hasil Eksperimen)

```json
{
  "split_ratio": 0.8,
  "neutral_threshold": 1.0,
  "tfidf_max_features": 5000,
  "tfidf_ngram_range": [1, 2],
  "tfidf_sublinear_tf": true,
  "tfidf_min_df": 2,
  "tfidf_max_df": 0.9,
  "svm_kernel": "linear",
  "svm_C": 10.0,
  "svm_gamma": "scale",
  "svm_class_weight": {
    "Positif": 1.0,
    "Netral": 1.0,
    "Negatif": 3.0
  }
}
```

### 6.3 Mengubah Konfigurasi

**Cara 1: Edit best_config.json**
```json
{
  "svm_C": 5.0,
  "svm_kernel": "rbf"
}
```

**Cara 2: Via Python**
```python
from scripts.config_loader import load_best_config, save_best_config

config = load_best_config()
config['svm_C'] = 5.0
config['svm_kernel'] = 'rbf'
save_best_config(config)
```

### 6.4 Parameter Detail

#### Split Configuration
| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| split_ratio | 0.8 | 80% train, 20% test |
| random_state | 42 | Untuk reproducibility |
| stratify | True | Balance kelas di split |

#### TF-IDF Configuration
| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| max_features | 5000 | Jumlah fitur maksimal |
| ngram_range | (1, 2) | Unigram + Bigram |
| sublinear_tf | True | Menggunakan log(1+tf) |
| min_df | 2 | Minimum dokumen frekuensi |
| max_df | 0.9 | Maksimum dokumen frekuensi |

#### SVM Configuration
| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| kernel | linear | Jenis kernel |
| C | 10.0 | Regularization parameter |
| gamma | scale | Kernel coefficient |
| class_weight | {"Negatif": 3.0} | Bobot kelas |

---

## 7. Pipeline Details

### 7.1 Preprocessing 7 Tahap

```
Tahap 1: Cleaning
  → Hapus emoji, URL, angka, tanda baca

Tahap 2: Case Folding
  → Ubah ke huruf kecil

Tahap 3: Tokenizing
  → Pecah teks menjadi token

Tahap 4: Normalisasi Slang
  → Ganti slang dengan kata baku
  → Menggunakan slang_dict_fixed.json

Tahap 5: Negation Handling
  → Gabungkan tidak+bagus → tidak_bagus
  → Daftar negasi: tidak, bukan, jangan, tak, kurang, dll

Tahap 6: Stopword Removal
  → Hapus kata umum (Sastrawi + tambahan)

Tahap 7: Stemming
  → Sastrawi Stemmer
  → Kurangi kata ke bentuk dasar
```

**Contoh Transformasi:**
```
"Aplikasi tidak bisa dibuka, error terus! 😞"

1. Cleaning:    "Aplikasi tidak bisa dibuka error terus"
2. Case Fold:   "aplikasi tidak bisa dibuka error terus"
3. Tokenize:     ["aplikasi", "tidak", "bisa", "dibuka", "error", "terus"]
4. Normalize:    ["aplikasi", "tidak", "bisa", "dibuka", "error", "terus"]
5. Negation:     ["aplikasi", "tidak_bisa", "dibuka", "error", "terus"]
6. Stopword:     ["aplikasi", "tidak_bisa", "dibuka", "error", "terus"]
7. Stem:         ["aplikasi", "tidak_bisa", "buka", "error", "terus"]

Hasil: "aplikasi tidak_bisa buka error terus"
```

### 7.2 Pelabelan InSet Lexicon

#### Cara Kerja

```
skor_total = Σ(skor kata positif) + Σ(skor kata negatif)

Klasifikasi:
  skor_total > 1.0  → Positif
  skor_total < -1.0 → Negatif
  -1.0 ≤ skor_total ≤ 1.0 → Netral
```

#### Contoh Perhitungan

```
Teks: "Aplikasi ini bagus dan membantu belajar"

Tokenisasi:
["aplikasi", "ini", "bagus", "dan", "membantu", "belajar"]

Skor:
- bagus: +4
- membantu: +4
- belajar: +0.5

skor_total = +4 + 4 + 0.5 = +8.5

Hasil: Positif (karena 8.5 > 1.0)
```

### 7.3 Deteksi Aspek

4 aspek yang dideteksi:

| Aspek | Kata Kunci |
|-------|------------|
| Kemudahan Akses | login, register, loading, wifi, koneksi, timeout |
| Gamifikasi | game, poin, ranking, leaderboard, level, xp, medal |
| Materi/Konten | materi, soal, quiz, kuis, pembahasan, jawaban |
| Performa Teknis | error, crash, lag, bug, hang, glitch, force close |

### 7.4 Evaluasi Model

#### Metrik yang Digunakan

| Metrik | Rumus | Target |
|--------|-------|--------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | > 85% |
| Precision | TP/(TP+FP) | > 80% |
| Recall | TP/(TP+FN) | > 80% |
| F1-Score | 2×(P×R)/(P+R) | > 85% |

#### Confusion Matrix

```
                  Prediksi
              Pos    Net    Neg
Aktual Pos   [ TP    FN     FN ]
       Net   [ FP    TP     FN ]
       Neg   [ FP    FP     TP ]
```

---

## 8. Troubleshooting

### 8.1 Error Umum

#### Error: Module Not Found

```bash
# Install dependencies
pip install pandas numpy scikit-learn Sastrawi google-play-scraper tqdm matplotlib
```

#### Error: Permission Denied

```bash
# Run as administrator (Windows) or with sudo (Linux)
# Atau gunakan virtual environment
```

#### Error: Data File Not Found

```bash
# Pastikan data ada di lokasi yang benar
ls data/pipeline_output/stage3_labeled.csv
```

### 8.2 Warning Umum

#### Warning: Low F1-Score

Penyebab:
1. Dataset tidak seimbang
2. Preprocessing terlalu agresif
3. Parameter tidak optimal

Solusi:
```python
# Gunakan class_weight untuk menangani imbalance
model = SVC(kernel='linear', C=10, class_weight={'Negatif': 3.0})

# Atau gunakan SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### Warning: Overfitting

```
Training F1: 0.99
Test F1: 0.85
Gap: 0.14 (> 0.10)
```

Solusi:
1. Tambah data training
2. Kurangi kompleksitas model (C lebih kecil)
3. Gunakan regularisasi
4. Kurangi max_features TF-IDF

### 8.3 Debugging Tips

#### Melihat Detail Preprocessing

```python
from shared.preprocessing import preprocess_text

text = "Aplikasi ini bagus tapi error terus!"
result, steps = preprocess_text(text, return_steps=True)

for stage, output in steps.items():
    print(f"{stage}: {output}")
```

#### Melihat Detail Pelabelan

```python
from shared.lexicon import classify_sentiment, calculate_sentiment_score

text = "Aplikasi ini bagus"
tokens = text.split()

print(f"Tokens: {tokens}")
print(f"Skor per token:")
for token in tokens:
    score = LEXICON.classify_token(token)
    print(f"  {token}: {score}")

print(f"Skor total: {calculate_sentiment_score(tokens)}")
print(f"Label: {classify_sentiment(text)['label']}")
```

#### Cek Konfigurasi Loaded

```python
from scripts.config_loader import load_best_config

config = load_best_config()
print(json.dumps(config, indent=2))
```

---

## 9. Referensi

### 9.1 File Penting

| File | Lokasi | Deskripsi |
|------|--------|-----------|
| best_config.json | output/experiments/ | Konfigurasi final |
| stage3_labeled.csv | data/pipeline_output/ | Data berlabel |
| metadata_latest.json | models/ | Metadata model |
| metodologi.md | docs/ | Dokumentasi metodologi |

### 9.2 API Reference

#### Preprocessing

```python
from shared.preprocessing import preprocess_text

# Preprocess satu teks
text = "Aplikasi ini bagus!"
processed = preprocess_text(text)
# Output: "aplikasi bagus"

# Preprocess dengan detail tahap
processed, steps = preprocess_text(text, return_steps=True)
```

#### Lexicon

```python
from shared.lexicon import classify_sentiment, get_label_distribution

# Klasifikasi satu teks
result = classify_sentiment("Aplikasi bagus", threshold=1.0)
# Output: {'label': 'Positif', 'raw_score': 4.0, ...}

# Distribusi label
distribution = get_label_distribution(df, 'sentiment_label')
```

#### Config Loader

```python
from scripts.config_loader import load_best_config, get_config

# Load semua config
config = load_best_config()

# Get satu parameter
kernel = get_config('svm_kernel')
C = get_config('svm_C')
```

### 9.3 Quick Commands

```bash
# Jalankan pipeline lengkap
python pipeline/run_pipeline.py

# Jalankan eksperimen
python scripts/run_experiments/run_experiments.py

# Test preprocessing
python -c "from shared.preprocessing import preprocess_text; print(preprocess_text('Test'))"

# Cek konfigurasi
python -c "from scripts.config_loader import load_best_config; print(load_best_config())"

# Training model
python pipeline/train_svm.py
```

---

## Changelog

| Tanggal | Versi | Perubahan |
|---------|-------|----------|
| 2024-01-15 | 1.0 | Initial release |

---

*Document ini adalah panduan resmi untuk menjalankan dan mengoptimisasi model Analisis Sentimen Wayground*
