import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from tests.conftest import PRODUCTION_DB


@pytest.fixture
def api():
    """
    Backend yang terisolasi ke database TEST.

    Environment variable (WAYGROUND_DATABASE_*) sudah di-set oleh
    tests/conftest.py SEBELUM backend di-import, sehingga engine SQLAlchemy
    tidak pernah menunjuk ke database produksi.
    """
    from backend import app as module
    from backend.database_models import db
    module.app.config['TESTING'] = True
    with module.app.app_context():
        assert Path(db.engine.url.database).resolve() != PRODUCTION_DB.resolve(), \
            'Test tidak boleh memakai database produksi.'
        db.drop_all()
        db.create_all()
    return module.app.test_client()


def seed(labels=True, legacy=False):
    from backend.app import app
    from backend.database_models import db, AnalysisSession, ReviewData
    with app.app_context():
        analysis = AnalysisSession(name='lifecycle test', filename='memory.csv', filepath='memory',
                                   original_count=3, has_actual_labels=labels, status='completed' if legacy else 'uploaded')
        db.session.add(analysis)
        db.session.flush()
        for text, label in [('Aplikasi error tidak bisa login', 'negatif'),
                            ('Sangat membantu belajar mengerjakan soal', 'positif'),
                            ('1234 !!!', 'netral')]:
            db.session.add(ReviewData(analysis_id=analysis.id, original_content=text,
                                      actual_label=label if labels else None,
                                      final_processed_text='error' if legacy else None,
                                      predicted_label='negatif' if legacy else None))
        db.session.commit()
        return analysis.id


def step(api, sid, name):
    result = api.post(f'/api/v1/analysis/{sid}/{name}')
    assert result.status_code == 200, result.json
    return result.json


def test_unlabeled_analysis_finishes_after_aspects_and_records_version(api):
    sid = seed(labels=False)
    for name in ['deduplicate', 'preprocess', 'predict', 'aspects']:
        step(api, sid, name)
    result = api.get(f'/api/v1/analysis/{sid}').json['analysis']
    assert result['status'] == 'completed'
    assert result['model_identity_status'] == 'recorded'
    assert result['model_version'] == '20261029_223454'
    assert result['prediction_count'] == 2
    assert result['aspects_ready'] and not result['evaluation_ready']
    evaluation = api.get(f'/api/v1/analysis/{sid}/evaluation').json
    assert evaluation['has_labels'] is False and evaluation['evaluation'] is None
    assert api.post(f'/api/v1/analysis/{sid}/evaluate').status_code == 400


def test_evaluation_then_aspects_stays_completed_and_reprediction_invalidates(api):
    sid = seed()
    for name in ['deduplicate', 'preprocess', 'predict', 'evaluate']:
        step(api, sid, name)
    assert api.get(f'/api/v1/analysis/{sid}').json['analysis']['status'] == 'evaluated'
    for _ in range(2):
        step(api, sid, 'aspects')
        assert api.get(f'/api/v1/analysis/{sid}').json['analysis']['status'] == 'completed'
    evaluation = api.get(f'/api/v1/analysis/{sid}/evaluation').json['evaluation']
    assert evaluation['model_version'] == '20261029_223454'
    assert evaluation['confusion_matrix_labels'] == ['negatif', 'netral', 'positif']
    assert sum(map(sum, evaluation['confusion_matrix'])) == 2
    step(api, sid, 'predict')
    current = api.get(f'/api/v1/analysis/{sid}').json['analysis']
    assert current['status'] == 'predicted'
    assert not current['aspects_ready'] and not current['evaluation_ready']
    assert api.get(f'/api/v1/analysis/{sid}/evaluation').json['evaluation'] is None
    assert not api.get(f'/api/v1/analysis/{sid}/aspects').json['data']


def test_upstream_rerun_clears_stale_predictions_and_identity(api):
    sid = seed()
    for name in ['deduplicate', 'preprocess', 'predict', 'aspects', 'evaluate']:
        step(api, sid, name)
    step(api, sid, 'preprocess')
    current = api.get(f'/api/v1/analysis/{sid}').json['analysis']
    assert current['model_version'] is None and not current['predictions_ready']
    assert current['preprocessing_ready']
    step(api, sid, 'deduplicate')
    assert not api.get(f'/api/v1/analysis/{sid}').json['analysis']['preprocessing_ready']


def test_legacy_model_version_is_unknown_until_explicit_rerun(api):
    sid = seed(legacy=True)
    current = api.get(f'/api/v1/analysis/{sid}').json['analysis']
    assert current['model_version'] is None and current['model_identity_status'] == 'unknown'
    assert api.post(f'/api/v1/analysis/{sid}/predict').status_code == 409
    for name in ['deduplicate', 'preprocess', 'predict']:
        step(api, sid, name)
    assert api.get(f'/api/v1/analysis/{sid}').json['analysis']['model_identity_status'] == 'recorded'


def test_pagination_and_missing_analysis_are_validated(api):
    sid = seed()
    result = api.get(f'/api/v1/analysis/{sid}/original?page=0&limit=0').json
    assert result['pagination']['page'] == result['pagination']['limit'] == 1
    assert len(result['data']) == 1
    assert api.get('/api/v1/analysis/9999999/original').status_code == 404


def test_additive_schema_migration_preserves_legacy_rows_and_backups(tmp_path):
    from flask import Flask
    from backend.database_models import db, AnalysisSession, ensure_analysis_schema
    app = Flask('stage5-migration-test')
    path = tmp_path / 'legacy.sqlite'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{path}'
    db.init_app(app)
    with app.app_context():
        for table in db.metadata.sorted_tables:
            if table.name != 'analysis_executions':
                table.create(bind=db.engine, checkfirst=True)
        db.session.add(AnalysisSession(name='legacy preserved', filename='old.csv', filepath='old', original_count=17))
        db.session.commit()
        ensure_analysis_schema()
        ensure_analysis_schema()
        row = AnalysisSession.query.one()
        assert row.name == 'legacy preserved' and row.original_count == 17
        assert row.to_dict()['model_version'] is None
    backups = list((tmp_path / 'backups').glob('*.sqlite'))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as connection:
        assert connection.execute('select name, original_count from analysis_sessions').fetchone() == ('legacy preserved', 17)
        assert connection.execute("select count(*) from sqlite_master where name='analysis_executions'").fetchone()[0] == 0
