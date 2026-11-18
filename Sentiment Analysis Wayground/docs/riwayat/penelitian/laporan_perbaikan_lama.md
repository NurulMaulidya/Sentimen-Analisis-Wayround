# Laporan Perbaikan Project Skripsi
## Analisis Sentimen Ulasan Play Store "Wayground" (TF-IDF + Linear SVM, 3 kelas, 4 aspek)

**Tanggal kerja**: 27 Oktober 2025
**Branch**: `fix/audit-perbaikan` (dari baseline `a272339`)
**Status**: Tahap 1–4 selesai, Tahap 2 **tertahan** di meja pemilik (anotasi belum diisi)

---

## 0. Ringkasan eksekutif

| Tahap | Status | Hasil utama |
|-------|--------|-------------|
| 1. Lexicon & pelabelan | ✅ selesai | Lexicon v2 (R3+R6) jadi default; akurasi vs 288 label pakar 0,8021 → **0,8194** |
| 2. Anotasi tambahan | ⏸ **tertahan** | Worksheet 500 baris siap; integrasi **ditolak** sampai anotasi diisi |
| 3. Dedup & split | ✅ selesai | Kebocoran 52 `processed_text` di dua split → **0** |
| 4. Preprocessing | ✅ selesai | Huruf berulang + emoji; 46 test preprocessing hijau |
| 5. Latih ulang | ⏳ lanjut | — |
| 6. Analisis aspek | ⏳ lanjut | — |
| 7. Kebersihan & test | ⏳ lanjut | — |

### Temuan yang bertentangan dengan brief (sudah diverifikasi ulang)

1. **Akurasi lexicon 0,569 tidak dapat direproduksi.** Nilai terukur **0,8021**.
   Angka yang mirip 57,3% adalah `label_changed: 57` pada
   `output/audit/expert_label_summary.json`.
2. **"771 ulasan Netral" tidak ada.** Terukur **562** (514 di antaranya rating 4-5).
3. **`scripts/audit_oov.py` tidak punya daftar OOV manual** seperti disebut brief.
   Yang ada: 5 kandidat hasil screening otomatis.
4. **"56 dari 612 holdout identik"** → terukur **52 teks duplikat** (69 baris,
   11,3%).
5. **Akar masalah kata sentimen "hilang"**: kata itu ada, tapi **bentuk stemmed**-nya
   tidak ada (`membantu` → `bantu`). 1.613 dari 8.079 entri lexicon (20%) tidak
   pernah bisa cocok; 626 entri frasa secara struktural mustahil cocok.

---

## 1. Tahap 1 — Lexicon dan pelabelan otomatis

### 1.1 Yang diubah

| File | Perubahan |
|------|-----------|
| `shared/lexicon.py` | Loader lexicon versi v1/v2; `classify_sentiment()` menjadi satu-satunya implementasi |
| `shared/labeling.py` | **Baru** — satu-satunya pintu masuk pelabelan (Stage 3 + `audit_oov`) |
| `shared/constants.py` | `DEFAULT_LEXICON_VERSION` (keputusan: **v2**) + env override `WAYGROUND_LEXICON_VERSION` |
| `pipeline/run_pipeline.py` | Stage 3 memakai `shared.labeling.label_text` (jalur tunggal) |
| `data/lexicon/v2/` | **Baru** — lexicon v2 + `changes.csv` (kata, bobot lama, bobot baru, alasan) |
| `data/lexicon/v2_brief_lengkap/` | Varian brief literal (R1+R3+R4+R6) |
| `data/lexicon/v2_tanpa_hapus_topic/` | Varian pembanding tanpa R1 |
| `scripts/build_lexicon_v2.py` | **Baru** — build deterministik |
| `scripts/evaluate_lexicon_v2.py` | **Baru** — evaluasi vs 288 label pakar |
| `scripts/ablate_lexicon_rules.py` | **Baru** — abrasi tiap aturan |
| `docs/audit_lexicon.md` | **Baru** — laporan audit lexicon |

### 1.2 Aturan build

| Rule | Isi | Diterapkan di |
|------|-----|----------------|
| R1 | Hapus kata topik netral | `v2_brief_lengkap` |
| R2 | Periksa bobot janggal (`lambat` sudah −4) | semua varian |
| R3 | Kembalikan 39 kata sentimen yang hilang | semua varian |
| R4 | Tambah bentuk stemmed (dibatasi kamus independen) | brief & tanpa_hapus_topic |
| R6 | Pisahkan 625 entri frasa ke `phrases_*.tsv` | semua varian |

### 1.3 Hasil (reproducible)

Evaluasi terhadap 288 label pakar
(`output/lexicon_audit/lexicon_v2_evaluation.json`):

| Metode | Akurasi | F1-macro | F1-weighted |
|--------|---------|----------|-------------|
| Lexicon v1 (lama) | 0,8021 | 0,6020 | 0,7945 |
| **Lexicon v2 (R3+R6) — dipakai** | **0,8194** | **0,6137** | **0,8090** |
| v2 (R3+R4+R6, tanpa hapus topic) | 0,7639 | 0,5905 | 0,7508 |
| v2 brief literal (R1+R3+R4+R6) | 0,7535 | 0,5440 | 0,7402 |
| Baseline rating (≥4 pos, ≤2 neg, 3 netral) | 0,7847 | 0,5814 | 0,7942 |
| Kandidat gabungan v2 + rating | 0,8472 | 0,5841 | 0,8204 |

Ablasi: **R3 satu-satunya aturan yang menaikkan akurasi**; R1 (hapus kata
topik) −4 poin; R4 (bentuk stem) selalu menurunkan akurasi.

> Kandidat gabungan (v2+rating) tertinggi akurasinya, tetapi **tidak pernah
> memprediksi Netral** → tetap kandidat, belum jadi label final.

### 1.4 Keputusan pemilik

- **Lexicon v2 (R3+R6) dipakai sebagai default** (disetujui).
- **Perlakuan kata topik ditunda** → dipakai varian v2 (R3+R6) sementara;
  dicatat sebagai keterbatasan di Bab Pembahasan.

---

## 2. Tahap 2 — Validasi manual tambahan (TERTAHAN)

### 2.1 Worksheet anotasi

`dataset/anotasi_tambahan.xlsx` + `.tsv`, **500 baris**, `reviewId` unik semua.

| Stratum | Kuota | Jumlah |
|---------|-------|--------|
| A — semua Netral (lexicon v2) rating ≥ 4 | semua (wajib) | 342 |
| D — revalidasi dari 288 label experts lama | 40 | 40 |
| B — konflik label lexicon vs rating | 58 | 58 |
| C — acak per kelas | 20/kelas | 60 |

Kolom: `reviewId, content, processed_text, rating, label_lexicon_v2,
label_pakar_1, label_pakar_2, catatan, sumber_label_pakar_1, stratum,
alasan_sampling`.

`label_pakar_1` terisi 40 baris (diisi dari 288 label experts lama, **bukan
direkayasa**); `label_pakar_2` masih kosong.

### 2.2 Integrasi label (versi baru)

`pipeline/integrate_expert_labels.py` ditulis ulang:

- Gabung lewat `reviewId`; prioritas **experts > lexicon v2**.
- Pemetaan form lama → `reviewId` bertingkat:
  `content_exact` (288) → `content_prefix` (1) → `content_fuzzy` (4) =
  **293 dari 300** terpetakan (semula 288), sisanya tercatat `unmatched`.
- Kolom `label_source` (`Expert` / `Lexicon v2`) + `label_source_detail`.
- Cohen's kappa: antar-annotator (label_pakar_1 vs label_pakar_2) dan
  experts lama vs anotator baru.
- **Guard:** integrasi ditolak selama worksheet tidak berisi label baru.

### 2.3 Jejak sampel experts lama

**Tidak dapat ditelusuri** — tidak ada script/seed/catatan di repo maupun
`archive/`. Distribusi hasil menunjukkan sampel **tidak proporsional**
(rating 3: 10,9% sampel vs 3,2% dataset; rating 5: 58,0% vs 78,1%),
sepertinya kuota manual ke rating rendah. Dicatat sebagai keterbatasan.

### 2.4 Yang perlu dari pemilik

1. Mengisi `label_pakar_1` (500 baris) dan `label_pakar_2` (minimal 100 baris
   agar kappa bisa dihitung).
2. Setelah terisi: jalankan `python pipeline/integrate_expert_labels.py`.

Rincian: `docs/audit_annotation_sampling.md`.

---

## 3. Tahap 3 — Dedup dan split tanpa kebocoran

### 3.1 Yang diubah

| File | Perubahan |
|------|-----------|
| `shared/splitting.py` | **Baru** — `StratifiedGroupKFold` (grup = `processed_text`), `duplicate_report`, `assert_no_overlap` |
| `pipeline/run_pipeline.py` | Stage 5 memakai split berkelompok; keluaran ke `stage5_split_v2/`; delimiter CSV diseragamkan ke koma |
| `tests/test_split_leakage.py` | **Baru** — 8 test anti-kebocoran |

### 3.2 Angka sebelum/sesudah

| Metrik | Split lama | Split baru (berkelompok) |
|--------|-----------|--------------------------|
| Metode | `train_test_split` stratified per baris | `StratifiedGroupKFold` (grup = `processed_text`) |
| Development | 2.448 | 2.449 |
| Holdout | 612 | 611 |
| `processed_text` di kedua split | **52** | **0** |
| Baris holdout yang bocor | 69 (11,3%) | 0 |
| Jumlah grup | — | 2.829 |
| Grup duplikat | — | 107 |
| Baris di grup duplikat | — | 338 |
| Baris yang hilang bila dideduplikasi | — | 231 |
| Grup dengan label campuran | — | 0 |

Distribusi kelas holdout (label lama): Positif 357→373, Negatif 141→134,
Netral 114→104.

### 3.3 Keputusan desain

**Kelompok, bukan hapus.** Baris duplikat **tidak dihapus** (menghapus
231 baris = 7,6% data). Kebocoran dicegah dengan menaruh seluruh grup
duplikat pada satu split.

### 3.4 Test

`pytest tests/test_split_leakage.py -q` → **8 passed**.

---

## 4. Tahap 4 — Preprocessing

### 4.1 Yang diubah

| File | Perubahan |
|------|-----------|
| `shared/preprocessing.py` | Tambah `normalize_repeated_letters()` + `map_emojis()`; hapus blok `main()`/`_resolve_project_path()` duplikat (124 baris) |
| `tests/test_preprocessing.py` | Perbaiki key salah tulis `2_casefolding` → `2_case_folding`; tambah 2 kelas test baru (15 test) |
| `scripts/measure_preprocessing_impact.py` | **Baru** — ukur dampak perubahan ke dataset |

### 4.2 Normalisasi huruf berulang

Aturan:

1. Deretan **≥ 3** huruf sama → 1 huruf (selalu).
2. **Vokal ganda** → 1 vokal, kecuali kata sah
   (`saat`, `laat`, `taat`, `faat`, `haa`, `laa`, `saa`, `kaa`).
3. **Konsonan ganda** → 1 konsonan **hanya bila** bentuk tunggalnya ada di
   kosakata acuan (slang dict + normalization dict + lexicon).

| Masukan | Hasil | Catatan |
|------|-------|---------|
| `bangett` | `banget` | bentuk tunggal ada di kamus |
| `seruu` | `seru` | aturan vokal ganda |
| `sekalii` | `sekali` | aturan vokal ganda |
| `gacorrrrr` | `gacor` | deretan ≥ 3 |
| `saat` | `saat` | kata sah, tidak berubah |
| `laa` | `laa` | kata sah, tidak berubah |
| `banyak` | `banyak` | tidak ada huruf kembar |

Kata tanpa bentuk dasar yang dikenal (`banyakk`) **dibiarkan** — lebih aman
daripada merusak kata sah.

### 4.3 Emoji

**Keputusan: emoji tidak dibuang, tetapi dipetakan menjadi token sentimen**
yang bertahan sampai stemming dan menjadi fitur TF-IDF:

| Emoji | Token |
|-------|-------|
| 👍 ❤️ 💯 😄 😍 🥰 🤩 😢 😭 😡 👎 💔 | `emoji_positif` / `emoji_negatif` |
| 🤔 🤷 😕 | `emoji_netral` |

Emoji berulang identik diringkas jadi satu token. Emoji yang tidak ada di
peta tetap dibuang seperti sebelumnya.

> **Catatan penting:** token emoji **tidak** memengaruhi label lexicon
> (lexicon v2 tidak memuat token ini). Emoji hanya membantu model
> berbasis machine learning. Konsekuensinya, label otomatis dan label
> model bisa berbeda pada ulasan yang hanya berisi emoji — dicatat sebagai
> keterbatasan.

### 4.4 Test

`pytest tests/test_preprocessing.py -q` → **46 passed** (dari 30, +16 baru).
