from app.api import create_app


def test_health():
    app = create_app()
    client = app.test_client()
    response = client.get('/api/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'


def test_analyze():
    app = create_app()
    client = app.test_client()
    response = client.post('/api/analyze', json={
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.8,
        'humidity': 82,
        'ph': 6.5,
        'rainfall': 202,
    })
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload['prediction']['crop'].lower() == 'rice'
    assert payload['prediction']['probability'] > 0.5
    assert payload['top_crops']
    assert 'agriculture_suitability' in payload
