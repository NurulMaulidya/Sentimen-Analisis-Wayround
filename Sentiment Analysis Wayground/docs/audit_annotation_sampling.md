# Audit Sampling Anotasi (Tahap 2)
**Tanggal**: 27 Oktober 2025
**Artefak**: `dataset/anotasi_tambahan.xlsx`, `dataset/anotasi_tambahan.tsv`,
`scripts/build_annotation_sheet.py`, `pipeline/integrate_expert_labels.py` (versi baru)

---

## 1. Ringkasan

| Item | Nilai |
|------|-------|
| Jumlah baris worksheet | 500 |
| Baris unik berdasarkan `reviewId` | 500 (tanpa duplikasi) |
| `label_pakar_1` terisi | 40 (diisi dari 288 label experts lama, stratum D) |
| `label_pakar_2` terisi | 0 — **menunggu anotator kedua** |
| Integrasi label | **BELUM dijalankan** (guard menolak, lihat bagian 5) |

---

## 2. Cara sampel 500 ulasan dipilih

Script: `scripts/build_annotation_sheet.py` (deterministik, `random_state=42`).

### 2.1 Strata

| Stratum | Kuota | Aturan | Jumlah |
|---------|-------|--------|--------|
| **A. `A_netral_rating_tinggi`** | semua (wajib) | label lexicon v2 = Netral **dan** rating ≥ 4 | 342 |
| **D. `D_revalidasi_lama`** | 40 | baris dari 288 baris yang pernah dilabeli experts lama | 40 |
| **B. `B_konflik_rating_label`** | 58 | label lexicon v2 bertentangan dengan label berbasis rating | 58 |
| **C. `C_acak_per_kelas`** | 20 per kelas | sampel acak dari sisa tiap kelas lexicon v2 | 60 |

Prioritas: **A > D > B > C**. Baris yang masuk stratum A/D tidak boleh
masuk lagi ke B/C agar tidak dianotasi dua kali.

### 2.2 Why A dominates

Brief memuat "WAJIB mencakup: semua Netral hasil lexicon yang berating >= 4".
Di dataset canonical ada **342** baris demikian — lebih dari separuh kuota.
Konsekuensinya worksheet ini **tidak proporsional terhadap dataset**:
kelas Netral 386 dari 500 baris (77%). Ini disengaja (untuk menguji
apakah kelas Netral memang bermakna), tetapi harus disebut eksplisit di
bab pembahasan.

Komposisi kelas di worksheet:

| Kelas lexicon v2 | Jumlah worksheet | Jumlah dataset |
|------------------|------------------|---------------|
| Netral | 386 | 388 |
| Positif | 58 | 2.041 |
| Negatif | 56 | 631 |

### 2.3 Kolom keluaran

`reviewId`, `content`, `processed_text`, `rating`, `label_lexicon_v2`,
`label_pakar_1`, `label_pakar_2`, `catatan`, `sumber_label_pakar_1`,
`stratum`, `alasan_sampling`.

`reviewId` selalu terisi — integrasi label **wajib** lewat `reviewId`,
tidak boleh mengandalkan pencocokan teks untuk file ini.

---

## 3. Jejak sampel experts lama (300 baris)

### 3.1 Apakah bisa ditelusuri?

**Tidak.** Tidak ada script (di repo maupun di `archive/`) yang membuat
`dataset/Hasil_Validasi_Pakar.tsv`. Yang ada hanyalah file form hasil
isi dan `pipeline/integrate_expert_labels.py` (versi lama) yang
mencocokkannya. Tidak ada `seed`, tidak ada log pemilihan, tidak ada
dokumen metode sampling.

**Ini dicatat sebagai keterbatasan** (lihat bagian 6).

### 3.2 Bukti bahwa sampel lama tidak proporsional

Bandingkan 293 baris yang berhasil dipetakan dengan dataset canonical:

| Rating | Proporsi sampel lama (n=293) | Proporsi dataset (n=3.060) |
|--------|------------------------------|-----------------------------|
| 1 | 19,1% | 11,3% |
| 2 | 4,8% | 2,0% |
| 3 | 10,9% | 3,2% |
| 4 | 7,2% | 5,4% |
| 5 | 58,0% | 78,1% |

Distribusi label experts lama (Positif 170 / Negatif 110 / Netral 20)
juga tidak proporsional terhadap dataset (Positif 58,6% / Negatif 23,0% /
Netral 18,4%).

Kesimpulan: sampel lama **sepertinya dipilih manual dengan kuota** yang
mendahyungkan rating rendah (1-3), bukan acak proporsional. Ini tidak
dapat dibuktikan lebih jauh tanpa catatan lapangan.

### 3.3 Perbaikan pemetaan ke reviewId

Versi lama hanya mencocokkan `Ulasan Asli` exact dan menghasilkan
**288 dari 300** baris terpetakan. Versi baru menambahkan dua tingkat
pemetaan (keduanya tercatat di
`output/audit/expert_review_id_mapping.csv`):

| Metode | Jumlah | Penjelasan |
|--------|--------|------------|
| `content_exact` | 288 | sama seperti versi lama |
| `content_prefix` | 1 | teks canonical diawali teks form (teks form terpotong ellipsis saat scraping) |
| `content_fuzzy` | 4 | Jaccard token ≥ 0,85 **dan** selisih ≥ 0,10 terhadap kandidat kedua (selisih aktual 0,67–0,76) |
| `unmatched` | 7 | 6 ulasan berbahasa Inggris + 1 ulasan yang tidak ada di snapshot dataset |

Total terpetakan: **293 dari 300** (+5 dibanding versi lama), seluruhnya
`reviewId` unik.

Sisa 7 baris **tidak** dipetakan paksa. Form experts memuat 6 ulasan
bahasa Inggris yang tidak ada pada snapshot `stage3_final_labeled.csv`
(versi lama juga tidak memetakannya), jadi labelnya hanya ada di form.

---

## 4. Cohen's kappa

Fungsi `cohen_kappa()` ada di `pipeline/integrate_expert_labels.py`
dan dihitung pada dua kondisi:

1. **Antar-annotator pada worksheet baru** — `label_pakar_1` vs
   `label_pakar_2` pada baris yang sama. Akan dihitung otomatis begitu
   worksheet terisi.
2. **Experts lama vs annotator baru** — label experts lama vs
   `label_pakar_1` pada 40 baris stratum D. Mengukur apakah anotator
   baru konsisten dengan anotasi experts lama.

Belum ada angka kappa karena worksheet belum terisi.

---

## 5. Status: TAHAP 2 BERHENTI DI SINI

`pipeline/integrate_expert_labels.py` (versi baru) **menolak berjalan**
jika worksheet anotasi baru tidak berisi satu pun label baru:

```
==============================================================================
INTEGRASI DITOLAK: worksheet anotasi baru belum terisi.
File : dataset/anotasi_tambahan.tsv
Isi kolom label_pakar_1 dan/atau label_pakar_2 terlebih dahulu,
lalu jalankan ulang script ini.
Tahap 2 berhenti di sini sampai anotasi selesai.
==============================================================================
```

Label `label_pakar_1` yang sudah terisi sejak awal (40 baris, diisikan
dari 288 label experts lama) **tidak** dihitung sebagai anotasi baru,
supaya guard tidak bisa ditembus.

Berkas canonical `data/pipeline_output/stage3_final_labeled.csv` **tidak
ditimpa**. Keluaran integrasi versi baru ditulis ke
`data/pipeline_output/stage3_final_labeled_v2.csv`.

---

## 6. Keterbatasan

1. **Proses sampling 300 baris lama tidak dapat ditelusuri** — tidak ada
   script, seed, atau catatan metode. Hanya bisa diobservasi dari
   distribusi hasilnya (lihat 3.2).
2. **Worksheet baru tidak proporsional terhadap dataset** karena stratum A
   (wajib) berisi 342 baris. Untuk estimasi performa per kelas, angka
   dari worksheet ini tidak boleh dipakai tanpa penyesuaian bobot.
3. **Satu annotator untuk 288 label lama.** Cohen's kappa historis tidak
   dapat dihitung; hanya bisa dihitung ke depan.
4. **7 baris experts lama tanpa `reviewId`** — tidak masuk dataset
   canonical, jadi tidak bisa dipakai sebagai ground truth model.
5. **Maksimal 500 baris** berarti selisih 1 baris pada kelas Negatif
   (n=56) sudah berarti 1,8 poin persentase; F1 kelas kecil sangat
   tidak stabil.
