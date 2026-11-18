# PERBAIKAN CLASS IMBALANCE - RINGKASAN SOLUSI

## 📊 STATUS SAAT INI

### Distribusi Label (Test Set)
| Kelas | Jumlah | Persentase | F1-Score |
|-------|--------|------------|----------|
| Positif | 342 | 42.5% | 0.88 |
| Netral | 442 | 54.9% | 0.89 |
| Negatif | 21 | **2.6%** | **0.67** |

**Masalah**: Kelas Negatif hanya 21 sampel (2.6%) → Performa rendah (F1=0.67)

---

## 🔧 SOLUSI YANG SUDAH DITERAPKAN

### 1. Gabungan Lexicon Negatif
```
negative.tsv         : ~335 kata
negative_extended.tsv: +334 kata
negative_fixed.tsv   : +44 kata
negative_critical.tsv : +50 kata
───────────────────────────────
TOTAL               : ~774 kata negatif
```

File disimpan: `data/lexicon/negative_final.tsv`

### 2. Threshold Optimal
```python
# constants.py
NEUTRAL_THRESHOLD = 0.8  # (dari 1.103)
```

---

## 📈 HASIL PERBAIKAN

### Perbandingan Threshold

| Threshold | Positif | Netral | Negatif | Catatan |
|-----------|---------|--------|---------|---------|
| T=1.103 (lama) | 42.2% | 55.1% | **2.7%** | Kurang negatif |
| T=1.0 | 43.8% | 53.2% | **3.0%** | Sedikit lebih baik |
| T=0.8 | 56.8% | 39.0% | **4.2%** | ✅ **Rekomendasi** |
| T=0.6 | 64.6% | 30.1% | **5.3%** | Agak terlalu banyak Positif |

---

## 🚀 CARA MENGGUNAKAN PERBAIKAN

### Opsi 1: Jalankan Ulang Pipeline
```bash
# Update threshold di shared/constants.py
# NEUTRAL_THRESHOLD = 0.8

# Jalankan ulang pipeline
python pipeline/run_pipeline.py
```

### Opsi 2: Relabel Data Existing
```bash
# Jalankan script perbaikan
python scripts/fix_imbalance.py

# Ini akan membuat file:
# data/pipeline_output/stage3_labeled_fixed.csv
```

### Opsi 3: Tambahan - SMOTE untuk Train
Jika masih imbalanced, gunakan SMOTE saat training:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Sekarang X_resampled punya distribusi lebih seimbang
```

---

## ⚠️ CATATAN PENTING

### Lexicon Still Has Issues
Beberapa kata bermakna negatif masih salah klasifikasi:

| Kata | Status di Kamus | Masalah |
|------|-----------------|---------|
| `jenuh` | Ada di POSITIF (+2.0) | Seharusnya negatif |
| `ngantuk` | Ada di POSITIF (+2.0) | Seharusnya negatif |
| `para` | Ada di POSITIF (+1.78) | Ambiguous |
| `lot` | Ada di POSITIF (+2.0) | Dari "lemot" |

**Solusi**: Edit file `data/lexicon/positive_extended.tsv` untuk memberikan skor 0 atau hapus kata-kata ini.

### Data Characteristics
Bila lexicon sudah diperbaiki, distribusi natural ulasan Play Store cenderung lebih positif karena:
- Pengguna puas lebih mungkin meninggalkan review
- Review negatif lebih panjang dan detail

---

## ✅ REKOMENDASI

1. **Jangka Pendek**: Turunkan threshold ke T=0.8
2. **Jangka Menengah**: Fix lexicon positive untuk kata ambigu
3. **Jangka Panjang**: Collect lebih banyak data negatif dari sumber lain

---

## 📁 File yang Dibuat

- `data/lexicon/negative_final.tsv` - Lexicon negatif gabungan
- `data/lexicon/negative_fixed.tsv` - Perbaikan tambahan
- `data/lexicon/negative_critical.tsv` - Kata kritis
- `scripts/fix_imbalance.py` - Script untuk perbaikan otomatis
