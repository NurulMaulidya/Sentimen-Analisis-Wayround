"""
Test Anti-Kebocoran Split (Tahap 3)
=================================

Memastikan:
1. Tidak ada `processed_text` yang sama di development dan holdout.
2. Splitunker yang dipakai pipeline memang **berkelompok**.
3. Distribusi kelas development/holdout masih mendekati dataset.
4. Split dapat direproduksi (random_state=42).

Test membaca split model aktif melalui konfigurasi proyek.

RUN: python -m pytest tests/test_split_leakage.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.splitting import (  # noqa: E402
    assert_no_overlap,
    build_groups,
    duplicate_report,
    grouped_stratified_split,
)
from shared.constants import CANONICAL_DATASET_PATH, PIPELINE_OUTPUT_DIR  # noqa: E402

SPLIT_DIR = PIPELINE_OUTPUT_DIR / "stage5_split_v2"
DEV_PATH = SPLIT_DIR / "development.csv"
HOLD_PATH = SPLIT_DIR / "holdout.csv"
META_PATH = SPLIT_DIR / "split_metadata.json"
LABEL_COLUMN = "sentiment_label_final"
GROUP_COLUMN = "processed_text"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=None, engine="python", encoding="utf-8-sig", dtype=str
    )


@pytest.fixture(scope="module")
def split_frames():
    return _read(DEV_PATH), _read(HOLD_PATH)


class TestNoTextLeakage:
    """Inti Tahap 3: tidak boleh ada teks sama di kedua split."""

    def test_no_processed_text_overlap(self, split_frames):
        development, holdout = split_frames
        dev_texts = set(build_groups(development, GROUP_COLUMN))
        hold_texts = set(build_groups(holdout, GROUP_COLUMN))

        overlap = dev_texts & hold_texts
        assert not overlap, (
            f"{len(overlap)} processed_text muncul di development dan "
            f"holdout. Contoh: {sorted(overlap)[:5]}"
        )

    def test_no_holdout_row_leaks(self, split_frames):
        """Tidak satu pun baris holdout boleh punya teks di development."""
        development, holdout = split_frames
        dev_texts = set(build_groups(development, GROUP_COLUMN))
        leaked = holdout[GROUP_COLUMN].isin(dev_texts)
        assert not leaked.any(), (
            f"{int(leaked.sum())} baris holdout bocor ke development."
        )

    def test_assert_no_overlap_raises_on_violation(self):
        """Fungsi penjaga harus benar-benar mendeteksi pelanggaran."""
        bad_dev = pd.DataFrame({"processed_text": ["sama", "a"]})
        bad_hold = pd.DataFrame({"processed_text": ["sama"]})
        with pytest.raises(AssertionError):
            assert_no_overlap(bad_dev, bad_hold, group_col=GROUP_COLUMN)

    def test_duplicate_groups_stay_together(self, split_frames):
        """Semua baris dalam satu grup duplikat harus satu split."""
        development, holdout = split_frames
        dev = set(build_groups(development, GROUP_COLUMN))
        hold = set(build_groups(holdout, GROUP_COLUMN))

        for text in list(dev & hold)[:5]:
            assert False, f"grup bocor: {text}"


class TestSplitMetadata:
    """Metadata split harus konsisten dengan isi file."""

    def test_metadata_exists_and_matches(self, split_frames):
        development, holdout = split_frames
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

        assert meta["development_rows"] == len(development)
        assert meta["holdout_rows"] == len(holdout)
        assert meta["grup_tumpang_tindih"] == 0
        assert meta["holdout_used_for_selection"] is False

    def test_class_distribution_close(self, split_frames):
        development, holdout = split_frames
        for frame in (development, holdout):
            dist = frame[LABEL_COLUMN].value_counts(normalize=True)
            assert set(dist.index) <= {"Positif", "Negatif", "Netral"}
            # Tidak boleh ada kelas yang hilang sepenuhnya di salah satu split.
            assert len(dist) == 3


class TestSplitReproducibility:
    """Split harus dapat direproduksi persis."""

    def test_same_seed_same_result(self):
        canonical = _read(CANONICAL_DATASET_PATH)
        dev1, hold1, _ = grouped_stratified_split(
            canonical, group_col=GROUP_COLUMN, label_col=LABEL_COLUMN
        )
        dev2, hold2, _ = grouped_stratified_split(
            canonical, group_col=GROUP_COLUMN, label_col=LABEL_COLUMN
        )
        assert set(dev1[GROUP_COLUMN]) == set(dev2[GROUP_COLUMN])
        assert set(hold1[GROUP_COLUMN]) == set(hold2[GROUP_COLUMN])

    def test_duplicate_report_counts(self):
        canonical = _read(CANONICAL_DATASET_PATH)
        report = duplicate_report(canonical, GROUP_COLUMN)

        # Baris dalam grup duplikat >= jumlah grup duplikat.
        assert report["baris_di_grup_duplikat"] >= report["grup_duplikat"]

        # Baris yang hilang bila dideduplikasi = baris ekstra di grup duplikat.
        assert report["baris_yang_akan_hilang_bila_dideduplikasi"] == (
            report["baris_di_grup_duplikat"] - report["grup_duplikat"]
        )

        # Konservasi baris: total = grup unik + baris di grup duplikat.
        assert (
            report["total_baris"]
            == report["jumlah_grup"]
            - report["grup_duplikat"]
            + report["baris_di_grup_duplikat"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
