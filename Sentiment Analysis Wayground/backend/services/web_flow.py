"""Stored inspection data for inference; never fits a vectorizer or classifier."""
import json
import numpy as np

from backend.database_models import db, ReviewData, ReviewFeatures, UploadRow
from shared.labels import distribution as label_distribution, normalize_label


def feature_details(vectorizer, row):
    names = vectorizer.get_feature_names_out()
    idf = (np.concatenate([part.idf_ for _, part in vectorizer.transformer_list])
           if hasattr(vectorizer, 'transformer_list') else vectorizer.idf_)
    features = [{'feature': str(names[i]), 'idf': float(idf[i]), 'weight': float(value)}
                for i, value in zip(row.indices, row.data)]
    features.sort(key=lambda f: (-f['weight'], f['feature']))
    return {'nonzero': len(features), 'status': 'ready' if features else 'zero_vector',
            'features': features, 'vocabulary_size': len(names)}


def fitted_features(pipeline):
    return pipeline.named_steps.get('tfidf', pipeline.named_steps.get('features'))


def transform_reviews(analysis, pipeline, metadata):
    reviews = ReviewData.query.filter_by(analysis_id=analysis.id, is_duplicate=False).order_by(ReviewData.id).all()
    vectorizer = fitted_features(pipeline)
    texts = [r.final_processed_text or '' for r in reviews]
    matrix = vectorizer.transform(texts) if texts else None
    ReviewFeatures.query.filter_by(analysis_id=analysis.id).delete()
    for index, review in enumerate(reviews):
        payload = feature_details(vectorizer, matrix.getrow(index))
        payload['configuration'] = metadata['model']['tfidf']
        if not texts[index].strip():
            payload['status'] = 'empty_processed'
        db.session.add(ReviewFeatures(review_id=review.id, analysis_id=analysis.id,
                                      model_version=metadata['model_version'], payload=json.dumps(payload)))
    db.session.flush()
    return reviews, matrix


def review_upload_details(review, context):
    if not context:
        return {}
    key = str(review.id)
    return {
        'data_origin': context.review_origins.get(key),
        'lexicon_label': context.lexicon_labels.get(key) if review.final_processed_text else None,
    }


def source_row_dict(row, review=None, context=None):
    review = review or (db.session.get(ReviewData, row.review_id) if row.review_id else None)
    status = 'empty' if not row.content.strip() else 'duplicate' if review and review.is_duplicate else 'valid'
    return {'id': row.id, 'review_id': row.review_id, 'row_number': row.row_number,
            'original_content': row.content, 'raw_label': row.raw_label,
            'raw_data': json.loads(row.raw_data), 'actual_label': normalize_label(row.raw_label),
            'label_status': 'empty' if row.raw_label is not None and not normalize_label(row.raw_label) else None,
            'data_origin': context.origins[row.row_number - 1] if context else None,
            'is_duplicate': bool(review and review.is_duplicate), 'validation_status': status,
            'reason': {'empty': 'Teks ulasan kosong', 'duplicate': 'Duplikat teks (strip dan huruf kecil)', 'valid': 'Ulasan valid'}[status]}


def session_summary(analysis):
    reviews = ReviewData.query.filter_by(analysis_id=analysis.id).all()
    active = [r for r in reviews if not r.is_duplicate]
    sources = UploadRow.query.filter_by(analysis_id=analysis.id).all()
    features = ReviewFeatures.query.filter_by(analysis_id=analysis.id).all()
    feature_values = [json.loads(f.payload) for f in features]
    distribution = {c: sum(r.predicted_label == c for r in active) for c in ['negatif', 'netral', 'positif']}
    origins = analysis.upload_context.origins if analysis.upload_context else []
    actual_labels = [s.raw_label for s in sources] if sources else [r.actual_label for r in reviews]
    return {'uploaded': analysis.original_count,
            'canonical_matches': {split: sum(split in origin for origin in origins)
                                  for split in ('development', 'holdout')} if analysis.upload_context else None,
            'actual_distribution': label_distribution(actual_labels),
            'missing_actual_labels': sum(normalize_label(v) is None for v in actual_labels) if analysis.label_column else None, 'source_rows_preserved': len(sources),
            'source_complete': len(sources) == analysis.original_count if sources else len(reviews) == analysis.original_count,
            'empty_source': sum(not s.content.strip() for s in sources), 'valid_source': len(reviews),
            'duplicates': sum(r.is_duplicate for r in reviews), 'after_dedup': len(active),
            'preprocessed': sum(r.final_processed_text is not None for r in active),
            'empty_processed': sum(r.final_processed_text == '' for r in active),
            'zero_vectors': sum(f['status'] == 'zero_vector' for f in feature_values),
            'feature_ready': sum(f['status'] == 'ready' for f in feature_values),
            'tfidf_ready': bool(analysis.execution and analysis.execution.preprocessing_signature and len(features) == len(active) and len(active)),
            'tfidf_model_version': features[0].model_version if features else None,
            'predicted': sum(distribution.values()), 'distribution': distribution,
            'valid_actual_labels': sum(bool(r.actual_label) for r in reviews),
            'evaluable': sum(bool(r.predicted_label and r.actual_label) for r in active),
            'predicted_without_label': sum(bool(r.predicted_label and not r.actual_label) for r in active),
            'with_aspects': sum(bool(r.predicted_label and json.loads(r.detected_aspects or '[]')) for r in active),
            'without_aspects': sum(bool(r.predicted_label and not json.loads(r.detected_aspects or '[]')) for r in active)}
