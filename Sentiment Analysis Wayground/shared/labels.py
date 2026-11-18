"""
Shared label helpers
====================

Single source of truth for normalising sentiment labels.

The research uses three classes: Positif, Netral, Negatif. Depending on the
source, a label may be written as ``Positif``, ``positif``, ``Positive``,
``pos``, ``1`` and so on. Every module that reads a label (backend, scripts,
evaluation) MUST use this module so the same label always maps to the same
class.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

# Canonical class order used for confusion matrices and classification reports.
CANONICAL_CLASSES: List[str] = ["negatif", "netral", "positif"]

# Human readable spelling used in the dataset and the reports.
DISPLAY_CLASSES: List[str] = ["Negatif", "Netral", "Positif"]

_ALIASES = {
    "positif": "positif",
    "positive": "positif",
    "pos": "positif",
    "p": "positif",
    "1": "positif",
    "netral": "netral",
    "neutral": "netral",
    "neu": "netral",
    "n": "netral",
    "0": "netral",
    "negatif": "negatif",
    "negative": "negatif",
    "neg": "negatif",
    "-1": "negatif",
}


def normalize_label(label: object) -> Optional[str]:
    """
    Normalise one label to its canonical lowercase form.

    Returns ``None`` when the label is empty, missing or unknown, so callers
    can decide explicitly how to treat unknown values.
    """
    if label is None:
        return None
    try:
        import pandas as pd

        if pd.isna(label):
            return None
    except (TypeError, ValueError):
        pass

    value = str(label).replace("﻿", "").strip().lower()
    if not value:
        return None
    return _ALIASES.get(value, value)


def is_known_label(label: object) -> bool:
    """True when the label maps to one of the three research classes."""
    return normalize_label(label) in CANONICAL_CLASSES


def normalize_label_or_raise(label: object, context: str = "label") -> str:
    """Normalise a label and raise a clear error when it is not supported."""
    value = normalize_label(label)
    if value is None:
        raise ValueError(f"{context} kosong.")
    if value not in CANONICAL_CLASSES:
        raise ValueError(
            f"{context} tidak dikenal: {label!r}. "
            f"Label yang didukung: {sorted(CANONICAL_CLASSES)}."
        )
    return value


def to_display(label: object) -> str:
    """Canonical class -> display spelling (``positif`` -> ``Positif``)."""
    value = normalize_label(label)
    if value is None:
        return ""
    for canonical, display in zip(CANONICAL_CLASSES, DISPLAY_CLASSES):
        if value == canonical:
            return display
    return str(label)


def distribution(labels: Iterable[object]) -> dict:
    """
    Class distribution over the three canonical classes.

    Always returns all three keys (missing classes count as 0) so charts and
    tables never silently drop a class.
    """
    counts = {label: 0 for label in CANONICAL_CLASSES}
    for raw in labels:
        value = normalize_label(raw)
        if value in counts:
            counts[value] += 1
    return counts


__all__ = [
    "CANONICAL_CLASSES",
    "DISPLAY_CLASSES",
    "normalize_label",
    "normalize_label_or_raise",
    "is_known_label",
    "to_display",
    "distribution",
]
