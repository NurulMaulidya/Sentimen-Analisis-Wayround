"""Lifecycle of an analysis and the identity of its stored predictions."""

import hashlib
import json

from backend.database_models import db, ReviewData, AnalysisExecution, ReviewFeatures, AnalysisRunSnapshot


def fingerprint_signature(fingerprint):
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode('utf-8')).hexdigest()


def invalidate_results(analysis, clear_preprocessing=False, keep_signature=False):
    if analysis.execution and analysis.execution.model_version:
        snapshot = {'analysis': analysis.to_dict(), 'reviews': [r.to_dict() for r in analysis.reviews],
                    'aspects': [a.to_dict() for a in analysis.aspects],
                    'evaluation': analysis.evaluation.to_dict() if analysis.evaluation else None,
                    'features': [{'review_id': f.review_id, 'model_version': f.model_version, 'data': json.loads(f.payload)} for f in analysis.feature_rows]}
        db.session.add(AnalysisRunSnapshot(analysis_id=analysis.id, payload=json.dumps(snapshot)))
    ReviewFeatures.query.filter_by(analysis_id=analysis.id).delete()
    signature = analysis.execution.preprocessing_signature if keep_signature and analysis.execution else None
    analysis.aspects.clear()
    analysis.evaluation = None
    analysis.execution = None
    for review in ReviewData.query.filter_by(analysis_id=analysis.id).all():
        review.predicted_label = None
        review.is_correct = None
        review.detected_aspects = None
        if clear_preprocessing:
            for field in ['cleaned_text', 'case_folded_text', 'tokenized_tokens', 'normalized_tokens',
                          'negation_tokens', 'stopwords_tokens', 'stemmed_tokens', 'final_processed_text']:
                setattr(review, field, None)
    db.session.flush()
    if signature:
        analysis.execution = AnalysisExecution(preprocessing_signature=signature)


def update_status(analysis):
    execution = analysis.execution
    if not execution or not execution.model_version:
        return
    if execution.aspects_ready:
        pending_evaluation = analysis.has_actual_labels and not execution.evaluation_ready and db.session.query(
            ReviewData.id
        ).filter(
            ReviewData.analysis_id == analysis.id,
            ReviewData.is_duplicate.is_(False),
            ReviewData.actual_label.isnot(None),
            ReviewData.predicted_label.isnot(None),
        ).first() is not None
        analysis.status = 'aspects_completed' if pending_evaluation else 'completed'
    elif execution.evaluation_ready:
        analysis.status = 'evaluated'
    else:
        analysis.status = 'predicted'
