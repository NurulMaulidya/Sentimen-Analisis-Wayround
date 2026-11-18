"""Normalize raw review columns and enforce the research period."""

import pandas as pd

from shared.constants import FINAL_OUTPUT_COLUMNS
from shared.research_config import RESEARCH_START, RESEARCH_END_EXCLUSIVE


def normalize_review_columns(frame):
    result = frame.copy()
    result.columns = [str(c).lstrip("\ufeff").strip() for c in result.columns]
    if result.columns.duplicated().any():
        raise ValueError("Duplicate column names after header normalization")
    if "tanggal" in result:
        if "at" in result:
            a = pd.to_datetime(result["at"], errors="coerce", utc=True)
            b = pd.to_datetime(result["tanggal"], errors="coerce", utc=True)
            if not a.dt.normalize().equals(b.dt.normalize()):
                raise ValueError("Conflicting at and tanggal columns")
            result = result.drop(columns="tanggal")
        else:
            result = result.rename(columns={"tanggal": "at"})
    missing = set(FINAL_OUTPUT_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"Missing review columns: {sorted(missing)}")
    result["at"] = pd.to_datetime(result["at"], errors="coerce", utc=True)
    if result["at"].isna().any():
        raise ValueError("Invalid or missing review date")
    if result["reviewId"].isna().any():
        raise ValueError("Missing reviewId")
    result["reviewId"] = result.reviewId.astype(str).str.strip()
    if result.reviewId.eq("").any() or result.reviewId.duplicated().any():
        raise ValueError("reviewId must be non-empty and unique")
    result["content"] = result.content.fillna("").astype(str)
    if result.content.str.strip().eq("").any():
        raise ValueError("Empty review content")
    ratings = pd.to_numeric(result.rating, errors="coerce")
    if not ratings.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Rating must be an integer from 1 to 5")
    result["rating"] = ratings.astype(int)
    return result


def filter_research_period(frame):
    frame = normalize_review_columns(frame)
    keep = frame["at"].ge(pd.Timestamp(RESEARCH_START, tz="UTC")) & frame["at"].lt(
        pd.Timestamp(RESEARCH_END_EXCLUSIVE, tz="UTC")
    )
    return frame.loc[keep].copy(), frame.loc[~keep].copy()
