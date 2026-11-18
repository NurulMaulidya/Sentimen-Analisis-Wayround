# Lexicon Wayground Versi 2

Folder ini berisi lexicon versi 2 beserta daftar perubahannya.
Lexicon versi 1 (`data/lexicon/lexicon_positive.tsv` dan
`lexicon_negative.tsv`) **tidak diubah dan tidak dihapus**.

## Berkas

| Berkas | Isi | Dipakai Stage 3? |
|--------|-----|------------------|
| `lexicon_positive.tsv` | 2.365 entri satu token, bobot positif | Ya |
| `lexicon_negative.tsv` | 5.126 entri satu token, bobot negatif | Ya |
| `phrases_positive.tsv` | entri frasa positif | Tidak |
| `phrases_negative.tsv` | entri frasa negatif | Tidak |
| `changes.csv` | daftar perubahan (kata, bobot lama, bobot baru, alasan) | — |

Format: TSV dengan header `word<TAB>weight`. Entri disortir bobot absolut
menurun lalu abjad.

## Varian yang tersedia

| Versi | Folder | Aturan |
|-------|--------|--------|
| `v1` | `data/lexicon/` | Lexicon lama (default pipeline) |
| `v2` | `data/lexicon/v2/` | **Rekomendasi**: R3 + R6 |
| `v2_brief_lengkap` | `data/lexicon/v2_brief_lengkap/` | R1 + R3 + R4 + R6 (brief literal) |
| `v2_tanpa_hapus_topic` | `data/lexicon/v2_tanpa_hapus_topic/` | R3 + R4 + R6 |

Memilih versi tanpa mengubah kode:

```bash
# Windows
set WAYGROUND_LEXICON_VERSION=v2
# Linux/macOS
export WAYGROUND_LEXICON_VERSION=v2
```

Nilai default ada di `shared/constants.py` (`DEFAULT_LEXICON_VERSION`).

## Aturan build

Dijalankan oleh `scripts/build_lexicon_v2.py` (deterministik, dapat diulang).

| Rule | Isi |
|------|-----|
| R1 | Hapus kata topik netral (`aplikasi`, `anak`, `soal`, `guru`, `sekolah`, `siswa`, `nilai`, `game`, `tarik`, `sama`, `buat`, `tambah`, `sering`, `lot`, + ekstensi) |
| R2 | Periksa bobot janggal |
| R3 | Kembalikan kata sentimen yang hilang (dari kamus + audit OOV + varian ejaan) |
| R4 | Tambah bentuk stemmed dari entri yang mati (dibatasi kamus independen) |
| R5 | Penjaga kata topik pada keluaran R4 |
| R6 | Pisahkan entri frasa ke `phrases_*.tsv` |

## Menguji ulang

```bash
python scripts/build_lexicon_v2.py --disable R1,R4   # varian rekomendasi
python scripts/evaluate_lexicon_v2.py                # evaluasi vs 288 label pakar
python scripts/ablate_lexicon_rules.py               # abrasi tiap aturan
```

Hasil dan analisis lengkap: `docs/audit_lexicon.md`.

## Peringatan

1. **Entri frasa tidak bisa cocok.** Classifier Stage 3 memecah teks dengan
   `split()` lalu mencocokkan satu token pada satu waktu. Karena itu file
   utama hanya berisi entri satu token, dan entri frasa dipisahkan.
2. **Bentuk kata harus stemmed.** `processed_text` adalah hasil stemming
   Sastrawi. Entri `"membantu"` tidak akan pernah cocok; yang cocok adalah
   `"bantu"`. Aturan R4 menambah bentuk stem, tetapi hanya bila kata
   stemmed-nya juga tercatat pada kamus sentimen independen — tanpa batas
   itu kualitas lexicon turun.
3. **Default pipeline masih `v1`.** Tidak ada dataset canonical yang
   ditimpa. Adopsi v2 sebagai sumber label menunggu persetujuan pemilik.
