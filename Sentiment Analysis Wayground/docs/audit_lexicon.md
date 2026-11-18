# Audit Lexicon Wayground

**Tanggal audit**: 27 Oktober 2025
**Ruang lingkup**: Tahap 1 perbaikan (lexicon & pelabelan otomatis)
**Artefak utama**: `data/lexicon/v2/`, `scripts/build_lexicon_v2.py`,
`scripts/evaluate_lexicon_v2.py`, `scripts/ablate_lexicon_rules.py`

---

## 1. Ringkasan Eksekutif

| No | Temuan | Status |
|----|--------|--------|
| F1 | Kata sentimen umum **tidak hilang** dari lexicon. Yang hilang adalah **bentuk stemmed**-nya (`membantu` → `bantu`) | Terverifikasi |
| F2 | **1.613 dari 8.078 entri (20,0%)** lexicon v1 **tidak pernah bisa cocok** dengan `processed_text` karena stemming Sastrawi | Terverifikasi |
| F3 | **626 entri frasa** (mengandung spasi) **secara struktural mustahil** dicocokkan oleh classifier berbasis token | Terverifikasi |
| F4 | OOV audit hanya menghasilkan **5 kandidat** (`bantu`, `keren`, `cocok`, `love`, `asyik`). 399 dari 779 kata kamus sentimen tidak ada di lexicon | Terverifikasi |
| F5 | Penghapusan kata topik (R1) dan penambahan bentuk stem massal (R4) **menurunkan** kecocokan dengan label pakar | Terverifikasi |
| F6 | Yang **memperbaiki** kualitas adalah pengembalian kata sentimen yang hilang (R3) | Terverifikasi |
| F7 | `scripts/audit_oov.py` **tidak lagi punya** daftar OOV manual seperti yang disebut pada brief | Terverifikasi |
| F8 | Angka "akurasi lexicon 0,569" dan "771 ulasan Netral" pada brief **tidak dapat direproduksi** dari artefak aktual | Terverifikasi |

---

## 2. Verifikasi terhadap konteks pada brief

Brief menyebut lima klaim. Semua diverifikasi ulang terhadap artefak aktual.

### Klaim 1 — "90,6% label (2.772 dari 3.060) dari lexicon InSet; hanya 288 label pakar"

**TERVERIFIKASI.** `data/pipeline_output/stage3_final_labeled.csv`:
`label_source` → `InSet: 2772`, `Expert: 288` (90,52% / 9,44%).

### Klaim 2 — "lexicon tidak memuat kata sentimen umum (bagus, bantu, seru, suka, senang, baik, mudah, keren)"

**TERVERIFIKASI SEBAGIAN, dan sebagian keliru.**(words | lexicon v1 | kamus)

| Kata | Lexicon v1 | kamus_positive | kamus_positif | Status |
|------|-------------|----------------|---------------|--------|
| bagus | **+2** | +4 | +1 | Sudah ada |
| bantu | — | +4 | — | **Hilang** |
| seru | **+5** | +4 | +1,5 | Sudah ada |
| suka | **+3** | +4 | +1 | Sudah ada |
| senang | **+5** | +4 | +1,5 | Sudah ada |
| baik | **+3** | +1 | +1 | Sudah ada |
| mudah | **+4** | +4 | +1 | Sudah ada |
| keren | — | +4 | +2 | **Hilang** |

Jadi hanya **4 dari 8** kata yang benar-benar hilang: `bantu`, `keren`, `cocok`, `love`
(plus `asyik` dari audit OOV). 
### Klaim 3 — "Kecocokan lexicon dengan pakar pada 288 baris itu 57,3%"

**TIDAK DAPAT DIREPRODUKSI.** Angka 57,3% tidak ditemukan pada artefak mana pun
(`docs/`, `output/`, `data/pipeline_output/`). Nilai yang terukur:

| Metode | Akurasi | F1-macro |
|--------|---------|----------|
| Lexicon v1 (tersimpan) vs 288 label pakar | **0,8021** | 0,6020 |
| Baseline rating vs 288 label pakar | 0,7847 | 0,5814 |

Angka yang mirip 57,3% justru adalah **"label_changed": 57** pada
`output/audit/expert_label_summary.json` — 57 dari 288 baris berubah labelnya
(19,8%). Ada pula `data/pipeline_output/labeling_validation_report.json` yang
menyimpan akurasi **0,7033** atas 300 baris, tetapi artefak itu **stale**: label
pakar di dalamnya berbeda (Netral 34, bukan 20) dan tidak cocok dengan file
pakar saat ini.

### Klaim 4 — "Kelas Netral = 'tidak ada kata lexicon' (771 ulasan, 712 berating 4-5)"

**TIDAK DAPAT DIREPRODUKSI.** Terverifikasi:

| Definisi | Jumlah Netral | Di antaranya rating 4-5 |
|----------|---------------|----------------------|
| `sentiment_label` aktual (skor = 0, threshold 0,5) | **562** | **514** |
| Varian ketat "tidak ada keyword lexicon" | 482 | 452 |
| Varian pada `dataset/stage3_final_labeled.csv` (versi lama) | 519 | — |

Angka 771 tidak muncul pada data maupun file lama mana pun.

### Klaim 5 — "56 dari 612 teks holdout (9,2%) identik dengan data development"

**TERVERIFIKASI dengan koreksi angka.** Terukur pada split saat ini:
**52** `processed_text` unik muncul di kedua split, sehingga **69 baris holdout
(11,3%)** terpengaruh. Angka 56/612 (9,2%) kemungkinan berasal dari
penghitungan versi berbeda. Detail dan penanganan ada di Tahap 3.

### Klaim tambahan — "audit_oov.py punya OOV manual (bagus, seru, bantu, ...)"

**TIDAK DAPAT DIREPRODUKSI.** `scripts/audit_oov.py` versi sekarang sudah
menyatakan `hard_coded_oov_active: False` dan **tidak memuat daftar OOV manual
yang dikodekan**. Daftar manual yang disebut brief tidak ditemukan di
`scripts/audit_oov.py` maupun di `archive/`.

Yang benar-benar ada adalah **5 kandidat hasil screening otomatis** di
`output/lexicon_audit/oov_kandidat_sentimen.tsv`:

| Kata | Frekuensi | Saran | Sumber |
|------|-----------|-------|--------|
| bantu | 339 | Positif 2 | kamus + screening |
| keren | 95 | Positif 2 | kamus + screening |
| cocok | 68 | Positif 2 | kamus + screening |
| love | 15 | Positif 2 | kamus + screening |
| asyik | 15 | Positif 2 | kamus + screening |

### Klaim tambahan — bobot kata topik

| Kata | Nilai pada brief | Nilai aktual di lexicon v1 | Tindakan |
|------|-------------------|------------------------------|---------|
| aplikasi | −4 | **tidak ada sama sekali** | Tetap tidak ada (sudah benar) |
| anak | −3 | −3 | Dihapus |
| soal | −1 | −1 | Dihapus |
| guru / sekolah / siswa | +3 | +3 | Dihapus |
| nilai | +5 | +5 | Dihapus |
| game | +2 | +2 | Dihapus |
| lambat | **+1** | **−4 (sudah negatif)** | Dipertahankan, dicatat |

---

## 3. Akar masalah: kata sentimen hilang

### 3.1 Bentuk stemmed, bukan kata yang hilang

Preprocessing Stage 2 menjalankan stemming Sastrawi sebagai **tahap terakhir**
(`shared/preprocessing.py`, `stem_tokens()`). Lexicon v1 disusun dari kata
**belum di-stem** (kamus InSet + form manual). Akibatnya, banyak entri lexicon
tidak pernah bisa cocok dengan `processed_text`.

```
Sastrawi: "membantu"  -> "bantu"
Lexicon v1 memuat: "membantu" (+4)     <- TIDAK PERNAH cocok
processed_text berisi: "bantu"          <- tidak ada di lexicon
```

**1.613 dari 8.078 entri (20,0%)** berada dalam kondisi ini. Contoh:

| Entri lexicon | Stem Sastrawi | Entri | Dampak |
|---------------|---------------|-------|--------|
| membantu (+4) | bantu | — | 339 kemunculan OOV |
| terindah (+5) | indah | — | kata pujian hilang |
| kekaguman (+4) | kagum | — | kata pujian hilang |
| interests / kepentingan (+5) | penting | — | kata netral jadi sentimen |
| keterbukaan (+5) | buka | — | kata netral jadi sentimen |
| bersengat (+5) | sengat | — | — |
|bertanggung (+4) | tanggung | — | — |

### 3.2 Entri frasa yang mustahil dicocokkan

`classify_sentiment()` memecah teks dengan `text.split()` lalu mencocokkan
**satu token** pada satu waktu. Entri yang mengandung spasi tidak pernah bisa
cocok:

| File | Entri frasa (tidak pernah cocok) |
|------|----------------------------------|
| `lexicon_positive.tsv` | 185 |
| `lexicon_negative.tsv` | 441 |
| **Total** | **626** |

Contoh: `kasih sayang`, `cinta kasih`, `terima kasih`, `sama sekali`,
`aplikasi jelek`, `sering error`. Semuanya tidak pernah pernah dipakai.

### 3.3 Kata sentimen yang hilang dari kamus

Dibandingkan dengan 4 file kamus (`kamus_positive.tsv`, `kamus_positif.tsv`,
`kamus_negative.tsv`, `kamus_negatif.tsv`): 779 kata unik, **399 kata (51%)**
tidak ada di lexicon.

Dari sisi negatif, 38 kata kamus negatif tidak ada di lexicon, tetapi hampir
semuanya kata kerja/noun deskriptif (`baca`, `bawa`, `besar`, `daftar`, `mulai`,
`selesai`, `tolong`, `mohon`, `habis`, `penting`, `segera`, `panjang`, `tinggi`).
Kata-kata ini **sengaja tidak ditambahkan** karena tidak membawa muatan
sentimen; menambahkannya akan menaikkan jumlah false positive.

---

## 4. Aturan build lexicon v2

Script: `scripts/build_lexicon_v2.py` (deterministik, dapat diulang).

| Rule | Isi | Jumlah perubahan | Diterapkan di |
|------|-----|------------------|----------------|
| **R1** | Hapus kata topik netral | 15 entri + 1 frasa | `v2_brief_lengkap` |
| **R2** | Periksa bobot janggal | 1 entri (nilai dipertahankan) | semua varian |
| **R3** | Kembalikan kata sentimen yang hilang | 39 entri | semua varian |
| **R4** | Tambah bentuk stemmed dari entri mati | 75 entri | `v2_brief_lengkap`, `v2_tanpa_hapus_topic` |
| **R5** | Penjaga kata topik pada keluaran R4 | 0 entri (R4 sudah menyaring) | — |
| **R6** | Pisahkan entri frasa ke `phrases_*.tsv` | 625 entri dipindah | semua varian |

Tiga varian yang dibangun (folder terpisah, **tidak menimpa v1**):

| Versi | Folder | Rule aktif | Entri positif | Entri negatif |
|-------|--------|-----------|----------------|----------------|
| `v1` | `data/lexicon/` | — (lama) | 2.511 | 5.568 |
| `v2` **(rekomendasi)** | `data/lexicon/v2/` | R3 + R6 | 2.365 | 5.126 |
| `v2_brief_lengkap` | `data/lexicon/v2_brief_lengkap/` | R1 + R3 + R4 + R6 | 2.395 | 5.156 |
| `v2_tanpa_hapus_topic` | `data/lexicon/v2_tanpa_hapus_topic/` | R3 + R4 + R6 | 2.407 | 5.159 |

Total entri v1 = **8.079**; total entri `v2` = **7.491**, semuanya satu token
(entri frasa sudah dipisah ke `phrases_*.tsv` sehingga bisa cocok).

### R1 — Daftar kata topik yang dihapus

Dari brief: `aplikasi`, `anak`, `soal`, `guru`, `sekolah`, `siswa`, `nilai`,
`game`, `tarik`, `sama`, `buat`, `tambah`, `sering`, `lot`.

Ekstensi (ditandai `topic_dihapus_ekstensi`, **perlu persetujuan**): `ajar`,
`ajarin`, `mengajar`, `ngajar`, `belajar`, `pelajaran`, `pembelajaran`,
`kuis`, `quiz`, `quizizz`, `quiziz`, `quizziz`, `wayground`, `main`, `app`,
`android`, `hp`. Hanya `belajar` (+3) dan `pelajaran` (+2) yang benar-benar
berada di lexicon v1.

### R3 — Kata yang dikembalikan

- **Dari audit OOV (5)**: `bantu` +3, `keren` +3, `cocok` +3, `love` +3,
  `asyik` +3.
- **Varian ejaan (20)**: `seruu`, `seruuu`, `baguss`, `bgus`, `bgs`, `bgtt`,
  `bangettt`, `bnget`, `sukaa`, `loveee`, `ngebantu`, `mantapp`, `mantappp`,
  `mntap`, `kerenn`, `kereen`, `kerennnn`, `kerennnnn`, `asyikk`, `asyikk`,
  `bett`.
- **Dari kamus (14)**: `best` +3, `gacor` +3, `enak` +3, `hebat` +3,
  `amazing` +3, `awesome` +3, `perfect` +3, `lumayan` +2, `pintar` +2,
  `lucu` +2, `super` +2, `giat` +2, `jos` +2, `gud` +2.

Bobot refresh 2 (bukan 3) dipakai untuk kata yang masih bersifat deskriptif
(`lumayan`, `pintar`, `lucu`, `super`, `giat`, `jos`, `gud`).

### R4 — Bentuk stemmed, dengan batas kamus independen

Awalnya R4 menambahkan **1.337** bentuk stem. Hasilnya **menurunkan** akurasi
dari 0,8021 menjadi 0,7431, karena banyak kata umum dari kamus InSet ikut
menjadi pemicu sentimen.

Batas yang dipakai: bentuk stem hanya ditambahkan bila kata stemmed-nya juga
tercatat pada kamus sentimen independen
(`kamus_positive/kamus_positif/kamus_negative/kamus_negatif`). Dasar ini
**independen dari label pakar**, sehingga tidak ada risiko overfitting ke 288
baris. Dengan batas ini hanya **75** bentuk stem yang masuk.

Selama build ditemukan **214 konflik bentuk stem** (dua entri berbeda
menghasilkan bentuk stemmed yang sama dengan polaritas berlawanan). Entri
pertama secara alfabetis yang dipakai, dan seluruh konflik dicatat di
`changes.csv` dengan kategori `konflik_bentuk_stem`.

> **Bug yang ditemukan & diperbaiki saat build**: versi pertama memakai
> snapshot `current` untuk pengecekan "sudah ada atau belum", sehingga
> 17 entri polaritasnya terbalik secara tidak-deterministik (mis. `bisa`
> menjadi −5 padahal +4). Diperbaiki dengan pengecekan langsung terhadap
> dict `positive`/`negative` dan iterasi alfabetis.

---

## 5. Hasil evaluasi terhadap 288 label pakar

Sumber: `scripts/evaluate_lexicon_v2.py` → `output/lexicon_audit/lexicon_v2_evaluation.json`

| Metode | Akurasi | F1-macro | F1-weighted |
|--------|---------|----------|-------------|
| Lexicon v1 (lama, default pipeline) | 0,8021 | 0,6020 | 0,7945 |
| **Lexicon v2 (R3 + R6) — rekomendasi** | **0,8194** | **0,6137** | **0,8090** |
| Lexicon v2 (R3 + R4 + R6, tanpa hapus topic) | 0,7639 | 0,5905 | 0,7508 |
| Lexicon v2 (R1+R3+R4+R6, brief literal) | 0,7535 | 0,5440 | 0,7402 |
| Baseline rating (≥4 pos, ≤2 neg, 3 netral) | 0,7847 | 0,5814 | 0,7942 |
| Kandidat gabungan v2 + rating | 0,8472 | 0,5841 | 0,8204 |

Distribusi label pada seluruh 3.060 baris:

| Metode | Positif | Negatif | Netral |
|--------|---------|---------|--------|
| Lexicon v1 (tersimpan) | 1.794 | 704 | 562 |
| **Lexicon v2 (R3 + R6)** | 2.041 | 631 | 388 |
| Lexicon v2 (R3+R4+R6, tanpa hapus topic) | 2.114 | 575 | 371 |
| Lexicon v2 (brief literal) | 2.086 | 570 | 404 |
| Baseline rating | 2.554 | 407 | 99 |

### 5.1 Ablasi tiap aturan

`scripts/ablate_lexicon_rules.py` → `output/lexicon_audit/lexicon_v2_ablation.json`

| Varian | Akurasi | F1-macro |
|--------|---------|----------|
| v1 (baseline, apa adanya) | 0,8021 | 0,6016 |
| v1 + R1 (hapus topic, daftar brief) | 0,7639 | 0,5571 |
| v1 + R1 (hapus topic, +ekstensi) | 0,7639 | 0,5571 |
| **v1 + R3 (kembalikan sentimen)** | **0,8194** | **0,6131** |
| v1 + R4 (bentuk stem, semua) | 0,7292 | 0,5232 |
| v1 + R4 (bentuk stem, freq≥5) | 0,7569 | 0,5683 |
| v1 + R3 + R4 (freq≥5) | 0,7674 | 0,5754 |
| v1 + R4 (kamus-grounded) | 0,7569 | 0,5904 |
| v1 + R3 + R4 (kamus-grounded) | 0,7674 | 0,5934 |
| v1 + R1 (brief) + R3 + R4 (kamus-grounded) | 0,7743 | 0,5566 |
| v1 + R1 (brief+ekst) + R3 + R4 (kamus-grounded) | 0,7535 | 0,5440 |
| **v2 (semua aturan, seperti dibangun)** | 0,7500 | 0,5154 |

**Kesimpulan abrasi:**

- **R3 (kembalikan kata sentimen)**: satu-satunya aturan yang menaikkan
  **akurasi** (0,8021 → 0,8194) dan F1-macro. Ini persis yang diminta brief.
- **R1 (hapus kata topik)**: menurunkan akurasi ~4 poin. Kata topic
  (`guru`, `sekolah`, `nilai`) memang tidak seharusnya membawa sentimen, tetapi
  pada 288 baris label pakar, kata topic itulah yang sering menjadi satu-satunya
  pemicu positif.
- **R4 (bentuk stem)**: **selalu menurunkan** akurasi. Bentuk stem yang
  ditambahkan dari kamus InSet banyak yang memicu false positive (kata umum
  ikut berlabel sentimen). Karena itu R4 tidak dipakai pada varian `v2`
  yang direkomendasikan.

---

## 6. Keterbatasan yang harus diketahui pembaca

1. **Label tidak sepenuhnya independen.** Kolom `Kesesuaian Label Awal` pada
   `dataset/Hasil_Validasi_Pakar.tsv` bernilai **"Sesuai" untuk 300 dari 300
   baris**. Artinya, annotatechno menyetujui label lexicon yang diuji. Kecocokan 0,8021 karena itu **tidak
   sepenuhnya independen**: pengukuran ini mengukur berapa jauh lexicon
   menyimpang dari label yang justru berbasis lexicon v1. Konsekuensinya,
   **penghapusan kata topik pasti menurunkan angka ini**, walaupun
   secara metodologis justru lebih benar.
2. **Ukuran sampel.** Hanya 288 baris (dari 300 pada form) yang bisa
   dicocokkan ke dataset canonical, dan hanya **19 baris** kelas Netral.
   Selisih 0,02 pada akurasi = sekitar 6 baris. Angka F1-macro Netral sangat
   tidak stabil.
3. **Hanya satu annotator.** Seluruh 300 baris berasal dari **satu orang**,
   bukan beberapa annotator. Cohen's kappa tidak dapat dihitung untuk label
   lama.
4. **Tidak ada reviewId pada form pakar.** Pencocokan lama memakai
   `Ulasan Asli` exact-match; 12 dari 300 baris tidak cocok.
5. **Label gabungan menghasilkan nol prediksi Netral.** Kandidat v2+rating
   attains akurasi tertinggi (0,8472) tetapi **tidak pernah memprediksi
   Netral**, sehingga kelas Netral pada model tidak akan pernah terpakai.
   Ini bukan perbaikan, melainkan perpindahan masalah.

---

## 7. Berkas yang dihasilkan

| Berkas | Isi |
|--------|-----|
| `data/lexicon/v2/lexicon_positive.tsv` | 2.365 entri satu token, positif |
| `data/lexicon/v2/lexicon_negative.tsv` | 5.126 entri satu token, negatif |
| `data/lexicon/v2/phrases_positive.tsv` | 184 entri frasa (tidak dipakai Stage 3) |
| `data/lexicon/v2/phrases_negative.tsv` | 441 entri frasa (tidak dipakai Stage 3) |
| `data/lexicon/v2/changes.csv` | Daftar perubahan: kata, polaritas/bobot lama, bobot baru, kategori, alasan |
| `data/lexicon/v2_brief_lengkap/` | Varian brief literal (R1+R3+R4+R6) |
| `data/lexicon/v2_tanpa_hapus_topic/` | Varian pembanding tanpa R1 |
| `scripts/build_lexicon_v2.py` | Script build (deterministik) |
| `scripts/evaluate_lexicon_v2.py` | Evaluasi vs 288 label pakar |
| `scripts/ablate_lexicon_rules.py` | Ablasi kontribusi tiap aturan |
| `shared/labeling.py` | **Satu-satunya** pintu masuk pelabelan |
| `shared/lexicon.py` | Loader lexicon, dukungan versi v1/v2 |
| `output/lexicon_audit/lexicon_v2_evaluation.json` | Hasil evaluasi |
| `output/lexicon_audit/lexicon_v2_ablation.json` | Hasil abrasi |

---

## 8. Keputusan yang menunggu persetujuan pemilik

1. **Adopsi lexicon v2 sebagai sumber label?** Default pipeline masih `v1`
   (`WAYGROUND_LEXICON_VERSION`). Tidak ada data canonical yang ditimpa.
2. **Kata topik dihapus atau dipertahankan?** (R1) Menghapus = lebih
   metodologis, akurasi −4 poin. Mempertahankan = akurasi lebih tinggi,
   tapi label tetap dipengaruhi kata topis.
3. **Label gabungan lexicon + rating?** Tidak dijadikan label final tanpa
   persetujuan; saat ini tidak memprediksi kelas Netral sama sekali.
4. **Ekstensi daftar topic** (`ajar`, `belajar`, `pelajaran`, `kuis`, `quiz`,
   `wayground`, `main`, `app`, `hp`) — perlu konfirmasi.
5. Kata deskriptif yang ditambahkan (`lumayan`, `pintar`, `lucu`, `super`,
   `giat`, `jos`, `gud`) — perlu konfirmasi.
