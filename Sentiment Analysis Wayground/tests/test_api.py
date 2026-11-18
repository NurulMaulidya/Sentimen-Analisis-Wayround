"""Test Flask API endpoints"""
import sys
import json
import pytest
sys.path.insert(0, '.')

from backend.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get('/api/v1/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('status') == 'healthy'


def test_config_endpoint(client):
    resp = client.get('/api/v1/config')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert data.get('split', {}).get('train_size') == 0.80
    assert data.get('split', {}).get('test_size') == 0.20
    assert data.get('svm', {}).get('kernel') == 'linear'
    assert data.get('svm', {}).get('C') == 10.0


def test_preprocess_endpoint(client):
    test_payload = json.dumps({'texts': ['Aplikasi bagus sekali', 'Error crash lag']})
    resp = client.post('/api/v1/preprocess',
                       data=test_payload,
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert len(data.get('results', [])) == 2


def test_label_endpoint(client):
    test_payload = json.dumps({'texts': ['bagus sekali', 'error crash lag', 'netral text']})
    resp = client.post('/api/v1/label',
                       data=test_payload,
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert len(data.get('results', [])) == 3


def test_stats_endpoint(client):
    resp = client.get('/api/v1/stats')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True


def test_enhanced_stats_endpoint(client):
    """
    Endpoint statistik dashboard memakai satu jalur yang sah.

    CATATAN: versi lama test ini memanggil /api/v1/enhanced/stats/dashboard,
    endpoint yang tidak pernah ada di backend dan tidak dipakai frontend.
    Statistik model final (metrik holdout terverifikasi) tersedia di
    /api/v1/model/stats, sedangkan statistik riwayat analisis di /api/v1/stats.
    """
    resp = client.get('/api/v1/model/stats')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'stats' in data
    assert data['stats']['integrity_verified'] is True
    # Metrik harus berasal dari artefak terverifikasi, bukan angka hard-coded.
    assert 0.0 <= data['stats']['accuracy'] <= 1.0
    assert data['stats']['metric_scope'] == 'diagnostic_holdout_with_prior_exposure'


def test_aspects_endpoint(client):
    test_payload = json.dumps({'texts': ['Aplikasi bagus tapi error login']})
    resp = client.post('/api/v1/aspects',
                       data=test_payload,
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert len(data.get('results', [])) > 0
