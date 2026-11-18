"""Deteksi aspek berbasis keyword dengan pencocokan token/phrase yang aman."""
from __future__ import annotations
import re
from typing import List, Dict, Tuple
import pandas as pd
try:
    from .constants import ASPECT_KEYWORDS, ASPECTS
except ImportError:
    from shared.constants import ASPECT_KEYWORDS, ASPECTS

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # Pertahankan kalimat yang terpisah oleh ., !, ?, atau newline.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]

def _keyword_pattern(keyword: str) -> re.Pattern:
    escaped = re.escape(str(keyword).strip().lower())
    # Boundary hanya di luar kata, sehingga phrase tetap dapat dicari.
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)

def detect_aspect(sentence: str, aspect_keywords: Dict[str, List[str]] | None = None) -> List[str]:
    if not isinstance(sentence, str) or not sentence.strip():
        return []
    keywords = ASPECT_KEYWORDS if aspect_keywords is None else aspect_keywords
    detected=[]
    text=sentence.lower()
    for aspect, words in keywords.items():
        for keyword in words:
            if _keyword_pattern(keyword).search(text):
                detected.append(aspect)
                break
    return detected

def detect_aspects_in_review(review_text: str, aspect_keywords=None) -> List[Dict]:
    results=[]
    for sentence in split_sentences(review_text):
        aspects=detect_aspect(sentence, aspect_keywords)
        if aspects:
            results.append({"sentence":sentence,"aspects":aspects})
    return results

def detect_all_aspects(text: str) -> Dict[str, List[str]]:
    result={aspect:[] for aspect in ASPECTS}
    result["Umum"]=[]
    for sentence in split_sentences(text):
        aspects=detect_aspect(sentence)
        if aspects:
            for aspect in aspects:
                result[aspect].append(sentence)
        else:
            result["Umum"].append(sentence)
    return result

def detect_aspect_batch(df: pd.DataFrame, text_column="content", review_id_column="reviewId") -> pd.DataFrame:
    for col in (text_column, review_id_column):
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan.")
    rows=[]
    for _, row in df.iterrows():
        for aspect, sentences in detect_all_aspects(row[text_column]).items():
            if aspect != "Umum":
                for sentence in sentences:
                    rows.append({review_id_column:row[review_id_column],"aspect":aspect,"sentence":sentence})
    return pd.DataFrame(rows, columns=[review_id_column,"aspect","sentence"])

def create_aspect_dataframe(df: pd.DataFrame, text_column="content", review_id_column="reviewId") -> Tuple[pd.DataFrame,pd.DataFrame]:
    aspects_df=detect_aspect_batch(df,text_column,review_id_column)
    if aspects_df.empty:
        return aspects_df, pd.DataFrame(columns=[review_id_column,"detected_aspects"])
    reviews=aspects_df.groupby(review_id_column)["aspect"].apply(lambda x:", ".join(sorted(set(x)))).reset_index()
    reviews.columns=[review_id_column,"detected_aspects"]
    return aspects_df,reviews

def add_aspects_to_dataframe(df: pd.DataFrame, text_column="content", review_id_column="reviewId") -> pd.DataFrame:
    result=df.copy()
    result["detected_aspects"]="Umum"
    for aspect in ASPECTS:
        result[f"has_{aspect.replace('/','_')}"]=False
    for idx,row in result.iterrows():
        detected=detect_all_aspects(row[text_column])
        names=[a for a in ASPECTS if detected.get(a)]
        result.at[idx,"detected_aspects"]=", ".join(names) if names else "Umum"
        for a in names:
            result.at[idx,f"has_{a.replace('/','_')}"]=True
    return result
