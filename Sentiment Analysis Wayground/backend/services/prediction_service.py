"""
Prediction Service
==================
Handles prediction using the pre-trained final_pipeline.joblib.
NO training or fitting - only inference.
"""

import warnings
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional

from shared.final_config import PROJECT_ROOT, FINAL_PIPELINE_PATH


# =============================================================================
# PIPELINE LOADING
# =============================================================================

def load_final_pipeline() -> Any:
    """
    Load the canonical final pipeline.

    Returns:
        sklearn.pipeline.Pipeline with TF-IDF + SVM

    Raises:
        FileNotFoundError: If pipeline not found
        TypeError: If loaded object is not valid
    """
    pipeline_path = Path(FINAL_PIPELINE_PATH)

    if not pipeline_path.is_absolute():
        pipeline_path = PROJECT_ROOT / pipeline_path

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Canonical final pipeline not found: {pipeline_path}"
        )

    from shared.deployment_integrity import load_verified_pipeline
    pipeline, _ = load_verified_pipeline()
    return pipeline


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single(text: str, pipeline: Any = None) -> Dict[str, Any]:
    """
    Predict sentiment for a single text.

    Args:
        text: Raw text to predict
        pipeline: Optional pre-loaded pipeline

    Returns:
        dict with prediction results
    """
    if pipeline is None:
        pipeline = load_final_pipeline()

    # Import preprocessing
    from .preprocessing_service import preprocess_full

    # Preprocess the text
    preprocessed = preprocess_full(text)
    final_text = preprocessed['final']
    if not final_text.strip():
        raise ValueError("Teks kosong setelah preprocessing")

    # Predict using the pipeline (handles TF-IDF transform internally)
    prediction = pipeline.predict([final_text])[0]
    label = _normalize_label(prediction)

    return {
        'original': text,
        'processed_text': final_text,
        'preprocessing_steps': preprocessed,
        'predicted_label': label,
        'sentiment_label': label,
        'model_version': pipeline._deployment_model_version
    }


def predict_batch(texts: List[str], pipeline: Any = None) -> List[Dict[str, Any]]:
    """
    Predict sentiment for multiple texts.

    Args:
        texts: List of raw texts
        pipeline: Optional pre-loaded pipeline

    Returns:
        List of prediction results
    """
    if pipeline is None:
        pipeline = load_final_pipeline()

    from .preprocessing_service import preprocess_full

    results = []
    for text in texts:
        result = predict_single(text, pipeline)
        results.append(result)

    return results


def predict_with_labels(texts: List[str], actual_labels: List[str],
                        pipeline: Any = None) -> Dict[str, Any]:
    """
    Predict and evaluate against actual labels.

    Args:
        texts: List of raw texts
        actual_labels: List of ground truth labels
        pipeline: Optional pre-loaded pipeline

    Returns:
        dict with predictions and evaluation metrics
    """
    if len(texts) != len(actual_labels):
        raise ValueError("Texts and labels must have same length")

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score
    )

    predictions = predict_batch(texts, pipeline)
    predicted_labels = [p['predicted_label'] for p in predictions]
    normalized_actual = [_normalize_label(l) for l in actual_labels]
    labels = ['negatif', 'netral', 'positif']
    if any(label not in labels for label in normalized_actual):
        raise ValueError("Label aktual tidak dikenal")

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(normalized_actual, predicted_labels)),
        'precision_macro': float(precision_score(normalized_actual, predicted_labels,
                                                  labels=labels, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(normalized_actual, predicted_labels,
                                            labels=labels, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(normalized_actual, predicted_labels,
                                     labels=labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(normalized_actual, predicted_labels,
                                        average='weighted', zero_division=0))
    }

    # Confusion matrix
    labels = ['negatif', 'netral', 'positif']
    cm = confusion_matrix(normalized_actual, predicted_labels, labels=labels)

    # Classification report
    report = classification_report(normalized_actual, predicted_labels,
                                    labels=labels, zero_division=0, output_dict=True)

    # Actual vs Predicted
    actual_vs_predicted = [
        {
            'actual': actual,
            'predicted': predicted,
            'correct': actual == predicted
        }
        for actual, predicted in zip(normalized_actual, predicted_labels)
    ]

    return {
        'predictions': predictions,
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'actual_vs_predicted': actual_vs_predicted,
        'correct_count': sum(1 for avp in actual_vs_predicted if avp['correct']),
        'incorrect_count': sum(1 for avp in actual_vs_predicted if not avp['correct'])
    }


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_label(label) -> str:
    """Normalize label ke kelas canonical (lowercase)."""
    from shared.labels import normalize_label

    return normalize_label(label)


def get_pipeline_info() -> Dict[str, Any]:
    """Get information about the loaded pipeline."""
    pipeline = load_final_pipeline()

    steps = []
    if hasattr(pipeline, 'steps'):
        steps = [name for name, _ in pipeline.steps]

    return {
        'path': str(FINAL_PIPELINE_PATH),
        'model_version': pipeline._deployment_model_version,
        'loaded': True,
        'pipeline_type': type(pipeline).__name__,
        'steps': steps,
        'has_predict': hasattr(pipeline, 'predict'),
        'has_predict_proba': hasattr(pipeline, 'predict_proba')
    }
