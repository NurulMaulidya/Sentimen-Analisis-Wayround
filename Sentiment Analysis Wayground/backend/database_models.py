"""
Database Models - Sentiment Analysis Wayground
==============================================
SQLAlchemy models for the redesigned sentiment analysis website.
"""

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def utcnow() -> datetime:
    """
    Waktu UTC naive, dipakai sebagai default kolom ``DateTime``.

    ``datetime.utcnow()`` sudah deprecated sejak Python 3.12, tetapi
    ``datetime.now(timezone.utc)`` menghasilkan objek yang timezone-aware.
    Kolom di sini-disimpan sebagai naive UTC dan serialized dengan akhiran
    ``Z`` (``to_dict``), jadi nilainya harus tetap naive UTC agar konsisten
    dengan baris lama dan dengan ``DateTime`` tanpa ``timezone=True``.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AnalysisSession(db.Model):
    """
    Main analysis session - tracks each upload/analysis.
    Each upload creates one session with all subsequent data linked to it.
    """
    __tablename__ = 'analysis_sessions'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    original_count = db.Column(db.Integer, default=0)
    dedup_count = db.Column(db.Integer, default=0)
    after_dedup_count = db.Column(db.Integer, default=0)
    has_actual_labels = db.Column(db.Boolean, default=False)
    label_column = db.Column(db.String(100))
    status = db.Column(db.String(50), default='uploaded')  # uploaded, deduplicated, preprocessing, analyzing, completed
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    reviews = db.relationship('ReviewData', backref='analysis_session', lazy=True, cascade='all, delete-orphan')
    aspects = db.relationship('AspectResult', backref='analysis_session', lazy=True, cascade='all, delete-orphan')
    evaluation = db.relationship('EvaluationResult', backref='analysis_session', uselist=False, cascade='all, delete-orphan')
    execution = db.relationship('AnalysisExecution', backref='analysis_session', uselist=False, cascade='all, delete-orphan')
    source_rows = db.relationship('UploadRow', cascade='all, delete-orphan')
    feature_rows = db.relationship('ReviewFeatures', cascade='all, delete-orphan')
    run_snapshots = db.relationship('AnalysisRunSnapshot', cascade='all, delete-orphan')

    upload_context = db.relationship('AnalysisUploadContext', uselist=False, cascade='all, delete-orphan')

    @property
    def mode(self):
        return self.upload_context.mode if self.upload_context else 'new_data'

    @property
    def is_research_demo(self):
        return self.mode == 'research_demo'

    @property
    def evaluation_scope(self):
        if not self.upload_context:
            return 'legacy_unknown'
        return 'research_descriptive' if self.is_research_demo else 'new_data'

    def to_dict(self):
        predicted = ReviewData.query.filter_by(analysis_id=self.id, is_duplicate=False).filter(ReviewData.predicted_label.isnot(None)).count()
        preprocessed = ReviewData.query.filter_by(analysis_id=self.id, is_duplicate=False).filter(ReviewData.final_processed_text.isnot(None)).count()
        execution = self.execution
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'mode': self.mode,
            'evaluation_scope': self.evaluation_scope,
            'evaluated_count': self.evaluation.correct_count + self.evaluation.incorrect_count if self.evaluation else 0,
            'preprocessing_signature': execution.preprocessing_signature if execution else None,
            'filepath': self.filepath,
            'original_count': self.original_count,
            'dedup_count': self.dedup_count,
            'after_dedup_count': self.after_dedup_count,
            'has_actual_labels': self.has_actual_labels,
            'label_column': self.label_column,
            'status': self.status,
            'model_version': execution.model_version if execution else None,
            'model_identity_status': 'recorded' if execution and execution.model_version else 'unknown' if predicted else 'not_predicted',
            'pipeline_sha256': execution.pipeline_sha256 if execution else None,
            'predicted_at': execution.predicted_at.isoformat() + 'Z' if execution and execution.predicted_at else None,
            'prediction_count': predicted,
            'distribution': {**dict.fromkeys(['negatif', 'netral', 'positif'], 0), **{label: count for label, count in db.session.query(ReviewData.predicted_label, db.func.count(ReviewData.id)).filter(
                ReviewData.analysis_id == self.id, ReviewData.is_duplicate == False, ReviewData.predicted_label.isnot(None)).group_by(ReviewData.predicted_label).all()}},
            'preprocessed_count': preprocessed,
            'dedup_ready': self.status != 'uploaded',
            'preprocessing_ready': preprocessed > 0,
            'predictions_ready': predicted > 0,
            'aspects_ready': bool(execution and execution.aspects_ready) or (not execution and (self.status == 'aspects_completed' or bool(self.aspects))),
            'evaluation_ready': bool(execution and execution.evaluation_ready) or bool(self.evaluation),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class AnalysisExecution(db.Model):
    """Identity of the current stored predictions, without altering legacy tables."""
    __tablename__ = 'analysis_executions'
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), primary_key=True)
    model_version = db.Column(db.String(100))
    pipeline_sha256 = db.Column(db.String(64))
    preprocessing_signature = db.Column(db.String(64))
    predicted_at = db.Column(db.DateTime)
    aspects_ready = db.Column(db.Boolean, default=False, nullable=False)
    evaluation_ready = db.Column(db.Boolean, default=False, nullable=False)


class ReviewData(db.Model):
    """
    Individual review with all preprocessing stages stored.
    Each review stores its state at every preprocessing step.
    """
    __tablename__ = 'review_data'

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False)
    original_content = db.Column(db.Text, nullable=False)

    # Preprocessing stages
    cleaned_text = db.Column(db.Text)
    case_folded_text = db.Column(db.Text)
    tokenized_tokens = db.Column(db.Text)  # JSON array
    normalized_tokens = db.Column(db.Text)  # JSON array
    negation_tokens = db.Column(db.Text)  # JSON array (after negation handling)
    stopwords_tokens = db.Column(db.Text)  # JSON array (after stopword removal)
    stemmed_tokens = db.Column(db.Text)  # JSON array (after stemming)
    final_processed_text = db.Column(db.Text)

    # Labels
    actual_label = db.Column(db.String(50))  # Ground truth if available
    predicted_label = db.Column(db.String(50))

    # Metadata
    is_duplicate = db.Column(db.Boolean, default=False)
    duplicate_of = db.Column(db.Integer, nullable=True)
    detected_aspects = db.Column(db.Text)  # JSON array
    is_correct = db.Column(db.Boolean, nullable=True)

    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self):
        import json
        result = {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'original_content': self.original_content,
            'cleaned_text': self.cleaned_text,
            'case_folded_text': self.case_folded_text,
            'tokenized_tokens': json.loads(self.tokenized_tokens) if self.tokenized_tokens else [],
            'normalized_tokens': json.loads(self.normalized_tokens) if self.normalized_tokens else [],
            'negation_tokens': json.loads(self.negation_tokens) if self.negation_tokens else [],
            'stopwords_tokens': json.loads(self.stopwords_tokens) if self.stopwords_tokens else [],
            'stemmed_tokens': json.loads(self.stemmed_tokens) if self.stemmed_tokens else [],
            'final_processed_text': self.final_processed_text,
            'actual_label': self.actual_label,
            'predicted_label': self.predicted_label,
            'is_duplicate': self.is_duplicate,
            'duplicate_of': self.duplicate_of,
            'detected_aspects': json.loads(self.detected_aspects) if self.detected_aspects else [],
            'is_correct': self.is_correct,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        return result

    def to_comparison_dict(self):
        """For compare view - shows all stages."""
        import json
        return {
            'id': self.id,
            'original': self.original_content,
            'cleaning': self.cleaned_text,
            'case_folding': self.case_folded_text,
            'tokenizing': json.loads(self.tokenized_tokens) if self.tokenized_tokens else [],
            'normalization': json.loads(self.normalized_tokens) if self.normalized_tokens else [],
            'negation': json.loads(self.negation_tokens) if self.negation_tokens else [],
            'stopwords': json.loads(self.stopwords_tokens) if self.stopwords_tokens else [],
            'stemming': json.loads(self.stemmed_tokens) if self.stemmed_tokens else [],
            'final': self.final_processed_text,
            'predicted_label': self.predicted_label,
            'actual_label': self.actual_label
        }


class AnalysisUploadContext(db.Model):
    __tablename__ = 'analysis_upload_contexts'

    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), primary_key=True)
    mode = db.Column(db.String(30), nullable=False)
    origins = db.Column(db.JSON, nullable=False)
    review_origins = db.Column(db.JSON, nullable=False)
    lexicon_labels = db.Column(db.JSON, nullable=False, default=dict)


class UploadRow(db.Model):
    """Original file rows, including those excluded from processing."""
    __tablename__ = 'upload_rows'
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False)
    review_id = db.Column(db.Integer, db.ForeignKey('review_data.id'))
    row_number = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    raw_label = db.Column(db.Text)
    raw_data = db.Column(db.Text, nullable=False)


class ReviewFeatures(db.Model):
    __tablename__ = 'review_features'
    review_id = db.Column(db.Integer, db.ForeignKey('review_data.id'), primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False)
    model_version = db.Column(db.String(100), nullable=False)
    payload = db.Column(db.Text, nullable=False)


class SinglePrediction(db.Model):
    __tablename__ = 'single_predictions'
    id = db.Column(db.Integer, primary_key=True)
    payload = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self):
        import json
        return dict(json.loads(self.payload), id=self.id, created_at=self.created_at.isoformat() + 'Z')


class AnalysisRunSnapshot(db.Model):
    __tablename__ = 'analysis_run_snapshots'
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False)
    payload = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)


class AspectResult(db.Model):
    """
    Aspect analysis results per review.
    """
    __tablename__ = 'aspect_results'

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False)
    review_id = db.Column(db.Integer, db.ForeignKey('review_data.id'), nullable=False)
    aspect_name = db.Column(db.String(100), nullable=False)
    sentences = db.Column(db.Text)  # JSON array of sentences
    # Sentimen ASPEK: dihitung dari kalimat-kalimat aspek itu sendiri
    # (lihat backend/services/aspect_service.py).
    sentiment = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self):
        import json
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'review_id': self.review_id,
            'aspect_name': self.aspect_name,
            'sentences': json.loads(self.sentences) if self.sentences else [],
            'sentiment': self.sentiment,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class EvaluationResult(db.Model):
    """
    Evaluation metrics and results.
    """
    __tablename__ = 'evaluation_results'

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_sessions.id'), nullable=False, unique=True)
    metrics = db.Column(db.Text)  # JSON: {accuracy, precision_macro, recall_macro, f1_macro, f1_weighted}
    classification_report = db.Column(db.Text)  # JSON classification report
    confusion_matrix = db.Column(db.Text)  # JSON 2D array
    actual_vs_predicted = db.Column(db.Text)  # JSON array of {actual, predicted, correct}
    correct_count = db.Column(db.Integer, default=0)
    incorrect_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self):
        import json
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'metrics': json.loads(self.metrics) if self.metrics else {},
            'classification_report': json.loads(self.classification_report) if self.classification_report else {},
            'confusion_matrix': json.loads(self.confusion_matrix) if self.confusion_matrix else [],
            'confusion_matrix_labels': ['negatif', 'netral', 'positif'],
            'model_version': self.analysis_session.execution.model_version if self.analysis_session.execution else None,
            'scope': self.analysis_session.evaluation_scope,
            'actual_vs_predicted': json.loads(self.actual_vs_predicted) if self.actual_vs_predicted else [],
            'correct_count': self.correct_count,
            'incorrect_count': self.incorrect_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


def ensure_analysis_schema():
    """Add new tables only; backup existing SQLite before the first addition."""
    import hashlib
    import json
    import sqlite3
    from contextlib import closing
    from pathlib import Path
    from sqlalchemy import inspect

    database = db.engine.url.database
    tables = inspect(db.engine).get_table_names()
    if db.engine.dialect.name == 'sqlite' and database and database != ':memory:' and tables and set(db.metadata.tables) - set(tables):
        path = Path(database).resolve()
        directory = path.parent / 'backups'
        directory.mkdir(exist_ok=True)
        stamp = utcnow().strftime('%Y%m%dT%H%M%S%fZ')
        backup = directory / f'{path.stem}_before_schema_update_{stamp}.sqlite'
        with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(backup)) as target:
            source.backup(target)
        (backup.with_suffix('.json')).write_text(json.dumps({
            'source': str(path), 'backup': str(backup), 'reason': 'Before additive tables: ' + ', '.join(sorted(set(db.metadata.tables) - set(tables))),
            'sha256': hashlib.sha256(backup.read_bytes()).hexdigest(), 'legacy_rows_rewritten': False,
        }, indent=2), encoding='utf-8')
    db.create_all()


def init_db(app):
    """Initialize database with app context."""
    db.init_app(app)
    with app.app_context():
        ensure_analysis_schema()


def get_db():
    """Get database instance."""
    return db
